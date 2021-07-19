from numpy import array, int32, float32, dtype, float64, ones, double
from h5py import h5t, h5f, h5d, h5s, File, Datatype
from typing import List

from src.utils.HDF5Utils import H5FileHandler

array_3_int32 = h5t.array_create(h5t.STD_I32LE, (3,))
array_7_float32 = h5t.array_create(h5t.IEEE_F32LE, (7,))


class H5CompoundType:
    def __init__(self, types: List, lengths: List[int], names: List[str]):
        """
        @param name: name of the type
        @param types: types within the compuond type
        @param names:  names of each type within the compound type
        """
        self.types = types
        self.names = names
        self.lengths = lengths
        self.create_type()

    def create_type(self):
        self.type = dtype([(name, t, (l,)) for name, t, l in zip(self.names, self.types, self.lengths)])


class DetPulseCoord(H5CompoundType):
    def __init__(self):
        super(DetPulseCoord, self).__init__([int32, float32], [3, 7], ["coord", "pulse"])


class WaveformPairNorm(H5CompoundType):
    def __init__(self):
        fields = ["t", "coord", "pulse", "phys", "EZ", "PID"]
        types = [double, int32, float32, float32, float32, int32]
        l = [1, 3, 130, 7, 2, 1]
        super(WaveformPairNorm, self).__init__(types, l, fields)


class WaveformNorm(H5CompoundType):
    def __init__(self):
        fields = ["t", "evt", "det", "pulse", "phys", "EZ", "PID"]
        types = [double, int32, int32, float32, float32, float32, int32]
        l = [1, 1, 1, 130, 7, 2, 1]
        super(WaveformNorm, self).__init__(types, l, fields)


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
