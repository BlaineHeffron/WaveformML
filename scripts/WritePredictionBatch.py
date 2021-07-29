import argparse
import subprocess
import os
import sys
from os.path import join

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.utils.util import check_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="path to input hdf5 files")
    parser.add_argument("mask", help="file mask (ie *WFNorm.h5)")
    parser.add_argument("config", help="path to config file for model")
    parser.add_argument("checkpoint", help="path to checkpoint file for model")
    parser.add_argument("--output", "-o", type=str, help="directory to output hdf5 files")
    parser.add_argument("--cpu", "-c", action="store_true", help="map tensor device storage to cpu")
    parser.add_argument("--num_threads", "-nt", type=int, help="number of threads to use")
    parser.add_argument("--buffer_size", "-b", type=int, help="number of rows to store in memory before writing to disk", default=1024*16)
    parser.add_argument("--read_size", "-r", type=int, help="number of rows per read", default=2048)

    args = parser.parse_args()
    input_path = check_path(args.input_path)
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    files = [join(input_path,f) for f in os.listdir(input_path) if f.endswith(args.mask[1:])]
    for f in files:
        args = ['python', join(dir_path, 'WritePredictions.py'), f, *sys.argv[3:]]
        print(" ".join(args))
        subprocess.call(args)

if __name__ == "__main__":
    main()