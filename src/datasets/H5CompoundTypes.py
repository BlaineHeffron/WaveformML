from numpy import array, int32, float32, dtype, float64
from h5py import h5t, h5f, h5d, h5s, File, Datatype
from typing import List

from src.utils.HDF5Utils import H5FileHandler

array_3_int32 = h5t.array_create(h5t.STD_I32LE, (3,))
array_7_float32 = h5t.array_create(h5t.IEEE_F32LE, (7,))


class H5CompoundType:
    def __init__(self, name : str, types : List[h5t.TypeID], names : List[str]):
        """
        @param name: name of the type
        @param types: types within the compuond type
        @param names:  names of each type within the compound type
        """
        self.name = name
        self.types = types
        self.names = names
        self.type = h5t.create(h5t.COMPOUND, self.calc_size())
        self.create_type()

    def calc_size(self):
        size = 0
        for type in self.types:
            size += type.get_size()
        return size

    def create_type(self):
        offset = 0
        for i, type in enumerate(self.types):
            print("{0} {1} {2}".format(self.names[i],offset,type))
            self.type.insert(self.names[i], offset, type)
            offset += type.get_size()


class DetPulseCoord(H5CompoundType):
    def __init__(self):
        super(DetPulseCoord, self).__init__(b"DetPulseCoord", [array_3_int32, array_7_float32], [b"coord", b"pulse"])


def main():
    a = DetPulseCoord()
    fileid = h5f.create(b"test.h5")
    x = [1,3,3]
    y = [1.,3.,3,4.,5,3.,33.]
    c = [(x,y)]
    data = {a.names[0]:x,a.names[1]:y}
    dspaceid = h5s.create_simple((1,),(h5s.UNLIMITED,))
    #dset = h5d.create(fileid, a.name, a.type, dspaceid)
    #dset.write()
    file = File("test.h5")
    numpytype = dtype([("coord",int32,(3,)),("pulse",float32,(7,))])
    data = array(c,dtype=numpytype)
    tid = h5t.C_S1.copy()
    tid.set_size(6)
    H5T6 = Datatype(tid)
    tid.set_size(4)
    H5T_C_S1_4 = Datatype(tid)
    file.create_dataset(a.name,data=data)
    file.attrs.create("CLASS","TABLE", dtype=H5T6)
    file.attrs.create("FIELD_0_NAME",a.names[0])
    file.attrs.create("FIELD_1_NAME",a.names[1])
    file.attrs.create("TITLE","Detpulse coord pair data")

    file.attrs.create("VERSION","3.0", dtype=H5T_C_S1_4)
    file.attrs.create("abstime",1.45e9,dtype=float64, shape=(1,))
    file.attrs.create("nevents",122421,dtype=float64, shape=(1,))
    file.attrs.create("runtime",125000,dtype=float64, shape=(1,))
    file.flush()



if __name__ == "__main__":
    main()
