import timeit
from numpy import append
from src.datasets.H5CompoundTypes import *
from src.utils.HDF5Utils import H5FileHandler
import os


class H5Base:

    def __init__(self, path, access='r', **kwargs):
        # self.h5f = tables.open_file(path, access, **kwargs)
        self.path = path
        self.h5f = H5FileHandler(path, access, **kwargs)

    '''
    def get_description(self, table_name):
        return self.h5f.get_node("/", table_name).description
    '''

    def close(self):
        self.h5f.close()


class H5Input(H5Base):

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.record_length = 0
        self.record_type = None
        self.table_name = ""
        self.table = None
        self.event_index_name = ""
        self.event_index_coord = None
        self.current_index = -1
        self.table_length = 0

    def setup_table(self, name, data_type, event_index_name, event_index_coord=None, base="/"):
        if self.table:
            self.table.close()
            self.table = None
        self.record_length = data_type.itemsize
        self.record_type = data_type
        self.table_name = name
        self.table = self.h5f[base + name]
        self.table_length = self.table.len()
        self.event_index_name = event_index_name
        self.event_index_coord = event_index_coord

    def get_event_number(self, row):
        if self.event_index_coord is None:
            return row[self.event_index_name]
        else:
            return row[self.event_index_name][self.event_index_coord]

    def next_chunk(self, nrows=2048, preserve_event=True):
        if not self.table:
            raise RuntimeError("No table opened!")
        if self.current_index == -2:
            self.current_index = -1
            return None
        if self.current_index == -1:
            self.current_index = 0
        if self.current_index + nrows >= self.table_length:
            self.current_index = -2
            return self.table[self.current_index:self.table_length]
        data = self.table[self.current_index:self.current_index + nrows]
        self.current_index += nrows
        if preserve_event:
            last_event = self.get_event_number(data[-1])
            current_row = self.table[self.current_index]
            while self.get_event_number(current_row) == last_event:
                data = append(data, current_row)
                self.current_index += 1
                if self.current_index >= self.table_length:
                    self.current_index = -2
                    return data
                current_row = self.table[self.current_index]
        return data


class H5Output(H5Base):
    def __init__(self, path):
        super().__init__(path, 'w')
        self.tables = {}
        self.table_index = {}

    def create_table(self, name, shape, data_type, compression="gzip", maxshape=(None,),
                     compression_opts=9, chunks=(1024,), **kwargs):
        self.tables[name] = self.h5f.create_dataset(name, shape=shape, dtype=data_type, compression=compression,
                                                    maxshape=maxshape, compression_opts=compression_opts, chunks=chunks,
                                                    **kwargs)
        self.table_index[name] = 0

    def add_rows(self, name, rows):
        self.tables[name][self.table_index[name]:self.table_index[name] + rows.shape[0]] = rows
        self.table_index[name] += rows.shape[0]

    def close_table(self, name):
        self.tables[name].close()
        self.table_index.pop(name)
        self.tables.pop(name)

    def flush(self, table=None):
        if table is None:
            self.h5f.flush()
        else:
            self.tables[table].flush()

    def copy_attrs(self, table, h5input, input_table, names, types, shapes):
        for n, t, s in zip(names, types, shapes):
            if n not in h5input.h5f[input_table].attrs.keys():
                print("Warning: {0} not in input table {1} attributes".format(n, input_table))
                continue
            if s is None:
                if t is None:
                    self.tables[table].attrs.create(n, h5input.h5f[input_table].attrs[n])
                else:
                    self.tables[table].attrs.create(n, h5input.h5f[input_table].attrs[n], dtype=t)
            else:
                if t is None:
                    self.tables[table].attrs.create(n, h5input.h5f[input_table].attrs[n], shape=s)
                else:
                    self.tables[table].attrs.create(n, h5input.h5f[input_table].attrs[n], dtype=t, shape=s)

    def copy_table(self, name, h5input):
        shape = h5input.h5f[name].shape
        dtype = h5input.h5f[name].dtype
        self.create_table(name, shape, dtype)
        if h5input.h5f[name].shape[0] > 0:
            self.tables[name].write_direct(h5input.h5f[name][()])


class P2XTableWriter(H5Output):
    def __init__(self, path):
        super(P2XTableWriter, self).__init__(path)

    def copy_chanmap(self, h5input):
        self.copy_table("Chanmap", h5input)
        self.copy_p2x_attrs(h5input, "Chanmap", "Chanmap")

    def get_attr_string_type(self, h5input, table, name):
        if name in h5input.h5f[table].attrs.keys():
            tid = h5t.C_S1.copy()
            tid.set_size(len(h5input.h5f[table].attrs[name]) + 1)
            return Datatype(tid)
        else:
            return None

    def copy_p2x_attrs(self, h5input, table, input_table):
        tid = h5t.C_S1.copy()
        tid.set_size(6)
        H5T6 = Datatype(tid.copy())
        names = ["CLASS"]
        shapes = [None]
        types = [H5T6]
        n = 0
        while "FIELD_{0}_NAME".format(n) in h5input.h5f[input_table].attrs.keys():
            names.append("FIELD_{0}_NAME".format(n))
            shapes.append(None)
            tid.set_size(len(h5input.h5f[input_table].attrs[names[-1]]) + 1)
            types.append(Datatype(tid.copy()))
            n += 1
        names.append("TITLE")
        shapes.append(None)
        tid.set_size(len(h5input.h5f[input_table].attrs[names[-1]]) + 1)
        types.append(Datatype(tid.copy()))
        names.append("VERSION")
        shapes.append(None)
        tid.set_size(len(h5input.h5f[input_table].attrs[names[-1]]) + 1)
        types.append(Datatype(tid.copy()))
        names.append("abstime")
        shapes.append((1,))
        types.append(float64)
        names.append("runtime")
        shapes.append((1,))
        types.append(float64)
        thistype = self.get_attr_string_type(h5input, input_table, "calgrp")
        if thistype is not None:
            names.append("calgrp")
            types.append(thistype)
            shapes.append(None)
        names.append("nevents")
        shapes.append((1,))
        types.append(float64)
        thistype = self.get_attr_string_type(h5input, input_table, "rname")
        if thistype is not None:
            names.append("rname")
            types.append(thistype)
            shapes.append(None)
        names.append("runtime")
        shapes.append((1,))
        types.append(float64)
        names.append("scalingfactor")
        shapes.append((1,))
        types.append(float64)
        self.copy_attrs(table, h5input, input_table, names, types, shapes)


def print_dtypes():
    myfiles = []
    for root, dirs, files in os.walk('/home/bheffron/projects/neutrino_ML/data/waveforms/type_rn'):
        # select file name
        for file in files:
            # check the extension of files
            if file.endswith('.h5'):
                # print whole path of files
                myfiles.append(os.path.join(root, file))
    # h5input = H5Input('/home/bheffron/projects/neutrino_ML/data/waveforms/type_rn'
    for f in myfiles:
        h5input = H5Input(f)
        name = "WaveformPairCal"
        if f.endswith("SE.h5"):
            name = "WaveformPairCal"
        elif f.endswith("WFNorm.h5"):
            name = "WaveformPairNorm"
        else:
            name = "WaveformNorm"
        # descr = h5input.get_description(name)
        # pt_dtype = dtype_from_descr(descr)
        # print(f)
        # print(pt_dtype)


def test1():
    # code snippet to be executed only once
    mysetup = '''from __main__ import H5Input
from src.datasets.H5CompoundTypes import WaveformPairNorm 
from numpy import empty
    '''

    mycode = ''' 
name = "/home/bheffron/projects/neutrino_ML/data/waveforms/type_i/s015_f00003_ts1520304327_WFNorm.h5"
h5input = H5Input(name)
data_type = WaveformPairNorm()
h5input.setup_table("WaveformPairNorm", data_type.type, "coord", 2)
data = h5input.next_chunk()
while data is not None:
    data = h5input.next_chunk()
    '''

    # timeit statement
    print(timeit.timeit(setup=mysetup,
                        stmt=mycode,
                        number=1))


def test2():
    # code snippet to be executed only once
    mysetup = '''
from __main__ import H5Input
from src.datasets.H5CompoundTypes import WaveformPairNorm 
from numpy import empty
    '''

    mycode = ''' 
name = "/home/bheffron/projects/neutrino_ML/data/waveforms/type_i/s015_f00003_ts1520304327_WFNorm.h5"
h5input = H5Input(name)
data_type = WaveformPairNorm()
h5input.setup_table("WaveformPairNorm", data_type.type, "coord", 2)
data = h5input.next_chunk(2048*4)
while data is not None:
    data = h5input.next_chunk(2048*4)
    '''

    # timeit statement
    print(timeit.timeit(setup=mysetup,
                        stmt=mycode,
                        number=1))



def main():
    # h5input = H5Input("/home/bheffron/projects/neutrino_ML/data/waveforms/ncapt/s015_f00002_ts1520300359_WFCalFiltered.h5")
    # test1()
    # test2()
    """
    pt_dtype = dtype_from_descr(descr)
    print("pytables descr")
    print(descr)
    wfpc = WaveformPairCal()
    print("my descr")
    print(wfpc.get_pytable_descr())
    print("pytables dtype")
    pt_dtype = dtype_from_descr(descr)
    print(dtype_from_descr(descr))
    print("my dtype")
    print(wfpc.type)
    print("pytables descr after using its dtype")
    print(descr_from_dtype(dtype_from_descr(descr)))

    new_type = dtype({'names': ['evt', 't', 'dt', 'z', 'E', 'PSD', 'PE', 'coord', 'waveform', 'EZ'],
                      'formats': ['<i8', '<f8', '<f4', '<f4', '<f4', '<f4', ('<f4', (2,)), ('<i4', (3,)),
                                  ('<i2', (130,)),
                                  ('<f4', (2,))], 'offsets': [0, 260, 272, 280, 284, 288, 292, 296, 304, 316],
                      'itemsize': 564})
    print("dtype from dict")
    print(new_type)
    print("desc from dict generated dtype")
    print(descr_from_dtype(new_type))
    h5out1 = H5Output("test1.h5")
    # h5out2 = H5Output("test2.h5")
    h5out1.create_table("WaveformPairCal", descr)
    # wfpc.type = new_type
    data = wfpc.generate_random_data(10)
    h5input.close()
    h5out1.add_rows("WaveformPairCal", data)
    h5out1.flush()
    h5out1.close()
    """

    # h5out2.create_table("WaveformPairCal", wfpc.get_pytable_descr())
    # print(data)
    # h5out1.add_rows("WaveformPairCal", data)
    # h5out2.add_rows(data)
    # h5out1.close()
    # h5out2.close()


if __name__ == "__main__":
    main()
