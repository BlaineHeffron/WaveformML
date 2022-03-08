from typing import List

from h5py import h5t, h5f, h5d, h5s, File, Datatype
from numpy import double, float32, int32, dtype, array, int64, int16, ones, float64
from numpy.random import randint


def extension_type_map(path):
    if path.endswith("WFNorm.h5"):
        return WaveformPairNorm()
    elif path.endswith("Phys.h5"):
        return PhysPulse()
    else:
        return WaveformPairCal()


class H5CompoundType:
    def __init__(self, types: List, lengths: List[int], names: List[str], name: str):
        """
        @param name: name of the type
        @param types: types within the compuond type
        @param names:  names of each type within the compound type
        """
        self.name = name
        self.types = types
        self.names = names
        self.lengths = lengths
        self.type = None
        self.size = 0
        self.offsets = []
        self.event_index_name = None
        self.event_index_coord = None
        self.calc_size()
        self.calc_offsets()
        self.create_type()

    def create_type(self):
        self.type = dtype([(name, t, (l,)) for name, t, l in zip(self.names, self.types, self.lengths)])

    def generate_random_data(self, length):
        arrays = {n: randint(0, high=5, size=(length, l) if l > 1 else (length,)).astype(t) for n, l, t in
                  zip(self.names, self.lengths, self.types)}
        # print(arrays)
        c = [tuple([arrays[name][i] for name in arrays]) for i in range(length)]
        # return array(arrays, dtype=self.type)
        return array(c, dtype=self.type)

    def calc_size(self):
        tot = 0
        for t, l in zip(self.types, self.lengths):
            tot += dtype(t).itemsize * l
        self.size = tot

    def calc_offsets(self):
        self.offsets = [0]
        for t, l in zip(self.types, self.lengths):
            self.offsets.append(self.offsets[-1] + dtype(t).itemsize * l)
        self.offsets = self.offsets[0:-1]


class DetPulseCoord(H5CompoundType):
    def __init__(self):
        super(DetPulseCoord, self).__init__([int32, float32], [3, 7], ["coord", "pulse"], "DetPulseCoord")


class WaveformPairNorm(H5CompoundType):
    def __init__(self):
        fields = ["t", "coord", "pulse", "phys", "EZ", "PID"]
        types = [double, int32, float32, float32, float32, int32]
        l = [1, 3, 130, 7, 2, 1]
        self.event_index_name = "coord"
        self.event_index_coord = 2
        super(WaveformPairNorm, self).__init__(types, l, fields, "WaveformPairNorm")

    """
    def create_type(self):
        self.type = dtype({'names': ['t', 'coord', 'pulse', 'phys', 'EZ', 'PID'],
                           'formats': ['<f8', ('<i4', (3,)), ('<f4', (130,)), ('<f4', (7,)), ('<f4', (2,)), '<i4'],
                           'offsets': [0, 520, 532, 560, 568, 572], 'itemsize': 1052})
    """

    def create_type(self):
        self.type = dtype({'names': ['t', 'coord', 'pulse', 'phys', 'EZ', 'PID'],
                           'formats': ['<f8', ('<i4', (3,)), ('<f4', (130,)), ('<f4', (7,)), ('<f4', (2,)), '<i4'],
                           'offsets': [560, 520, 0, 532, 572, 568], 'itemsize': 584})


class WaveformNorm(H5CompoundType):
    def __init__(self):
        fields = ["t", "evt", "det", "pulse", "phys", "EZ", "PID"]
        types = [double, int64, int32, float32, float32, float32, int32]
        l = [1, 1, 1, 130, 7, 2, 1]
        self.event_index_name = "evt"
        self.event_index_coord = None
        super(WaveformNorm, self).__init__(types, l, fields, "WaveformNorm")

    """
    def create_type(self):
        self.type = dtype({'names': ['t', 'evt', 'det', 'pulse', 'phys', 'EZ', 'PID'],
                           'formats': ['<f8', '<i8', '<i4', ('<f4', (59,)), ('<f4', (8,)), ('<f4', (2,)), '<i4'],
                           'offsets': [0, 236, 272, 280, 288, 296, 300], 'itemsize': 516})
    """


class WaveformPairCal(H5CompoundType):
    def __init__(self):
        fields = ["evt", "t", "dt", "z", "E", "PSD", "PE", "coord", "waveform", "EZ", "PID"]
        types = [int64, double, float32, float32, float32, float32, float32, int32, int16, float32, int32]
        l = [1, 1, 1, 1, 1, 1, 2, 3, 130, 2, 1]
        self.event_index_coord = 2
        self.event_index_name = "coord"
        super(WaveformPairCal, self).__init__(types, l, fields, "WaveformPairCal")


    def create_type(self):
        self.type = dtype({'names': ['evt', 't', 'dt', 'z', 'E', 'PSD', 'PE', 'coord', 'waveform', 'EZ', 'PID'],
                           'formats': ['<i8', '<f8', '<f4', '<f4', '<f4', '<f4', ('<f4', (2,)), ('<i4', (3,)),
                                       ('<i2', (130,)),
                                       ('<f4', (2,)), '<i4'],
                           'itemsize': 324})

class PhysPulse(H5CompoundType):
    def __init__(self):
        fields = ["evt", "seg", "E", "rand", "t", "dt", "PE", "y", "PSD", "PID", "E_SE", "Esmear_SE", "y_SE", "PSD_SE"]
        types = [int64, int32, float32, float32, double, float32, float32, float32, float32, int32, float32, float32,
                 float32, float32]
        l = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2]
        self.event_index_coord = 2
        self.event_index_name = "coord"
        super(PhysPulse, self).__init__(types, l, fields, "PhysPulse")

    def create_type(self):
        self.type = dtype({'names': ["evt", "seg", "E", "rand", "t", "dt", "PE", "y", "PSD", "PID", "E_SE", "Esmear_SE",
                                     "y_SE", "PSD_SE"],
                           'formats': ['<i8', '<i4', '<f4', '<f4', '<f8', '<f4', ('<f4', (2,)), '<f4', '<f4', '<i4',
                                       ('<f4', (2,)), ('<f4', (2,)), '<f4', ('<f4', (2,))],
                           'itemsize': 84})


class Waveform(H5CompoundType):
    def __init__(self):
        fields = ["evt", "det", "t", "a", "PSD", "waveform", "PID", "true_E", "true_Z"]
        types = [int64, int32, double, float32, float32, int16, int32, float32, float32]
        l = [1, 1, 1, 1, 1, 59, 1, 1, 1]
        self.event_index_name = "evt"
        super(Waveform, self).__init__(types, l, fields, "Waveforms")


def main():
    a = DetPulseCoord()
    fileid = h5f.create(b"test.h5")
    x = [1, 3, 3]
    y = [1., 3., 3, 4., 5, 3., 33.]
    x = ones((100, 3), dtype=int32)
    y = ones((100, 7), dtype=float32)
    z = ones((100, 2), dtype=float32)
    c = [(x[i], y[i], z[i]) for i in range(100)]
    data = {a.names[0]: x, a.names[1]: y}
    dspaceid = h5s.create_simple((1,), (h5s.UNLIMITED,))
    # dset = h5d.create(fileid, a.name, a.type, dspaceid)
    # dset.write()
    file = File("test.h5")
    numpytype = dtype([("coord", int32, (3,)), ("pulse", float32, (7,)), ("EZ", float32, (2,))])
    data = array(c, dtype=numpytype)
    tid = h5t.C_S1.copy()
    tid.set_size(6)
    H5T6 = Datatype(tid)
    tid.set_size(4)
    H5T_C_S1_4 = Datatype(tid)
    file.create_dataset("DetPulseCoord", data=data)
    file.attrs.create("CLASS", "TABLE", dtype=H5T6)
    file.attrs.create("FIELD_0_NAME", a.names[0])
    file.attrs.create("FIELD_1_NAME", a.names[1])
    file.attrs.create("TITLE", "Detpulse coord pair data")

    file.attrs.create("VERSION", "3.0", dtype=H5T_C_S1_4)
    file.attrs.create("abstime", 1.45e9, dtype=float64, shape=(1,))
    file.attrs.create("nevents", 122421, dtype=float64, shape=(1,))
    file.attrs.create("runtime", 125000, dtype=float64, shape=(1,))
    file.flush()


if __name__ == "__main__":
    main()
