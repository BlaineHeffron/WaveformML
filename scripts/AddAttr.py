import argparse
import h5py
from numpy import float64

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_file", help="file with correct runtime")
    parser.add_argument("file", help="file to add it")
    args = parser.parse_args()
    ref = h5py.File(args.ref_file, 'r')
    f = h5py.File(args.file, 'r+')
    f["PhysPulse"].attrs.create("runtime", ref["PhysPulse"].attrs["runtime"])
    f.flush()
    ref.close()
    f.close()


if __name__=="__main__":
    main()
