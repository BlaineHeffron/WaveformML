import argparse
from os.path import expanduser, isdir, join
from ntpath import basename
import torch

from src.datasets.PredictionWriter import *
from src.utils.util import check_path
import time
import tracemalloc


# ... run your application ...




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="path to input hdf5 file")
    parser.add_argument("config", help="path to config file for model")
    parser.add_argument("checkpoint", help="path to checkpoint file for model")
    parser.add_argument("--output", "-o", type=str, help="path to output hdf5 file")
    parser.add_argument("--cpu", "-c", action="store_true", help="map tensor device storage to cpu")
    parser.add_argument("--num_threads", "-nt", type=int, help="number of threads to use")

    args = parser.parse_args()
    input_path = check_path(args.input_path)
    config = check_path(args.config)
    checkpoint = check_path(args.checkpoint)
    print("Setting up output")
    output = input_path[0:-3] + "ModelOut.h5"
    if args.output is not None:
        out = expanduser(args.output)
        if out.endswith(".h5"):
            output = out
        elif isdir(out):
            fname = basename(input_path)
            output = join(out, fname[0:-3] + "ModelOut.h5")
        else:
            raise IOError("Output path {} not a valid directory or .h5 file".format(args.output))
    start_time = time.time()
    pw_args = {}
    if args.cpu:
        pw_args["map_location"] = "cpu"
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    tracemalloc.start()
    PW = ZPredictionWriter(output, input_path, config, checkpoint, **pw_args)
    print("Writing predictions")
    PW.write_predictions()
    runtime = time.time() - start_time
    first_size, first_peak = tracemalloc.get_traced_memory()
    print("first size {} ".format(first_size))
    print("first peak {} ".format(first_peak))
    print("Success")
    print("Writing XML metadata")
    PW.write_XML(runtime)


if __name__ == "__main__":
    main()