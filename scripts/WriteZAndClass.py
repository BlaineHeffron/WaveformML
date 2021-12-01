import argparse
from os.path import expanduser, isdir, join
from ntpath import basename
import sys, os
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.datasets.PredictionWriter import *
from src.utils.util import check_path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="path to input hdf5 file")
    parser.add_argument("config_z", help="path to config file for z model")
    parser.add_argument("checkpoint_z", help="path to checkpoint file for z model")
    parser.add_argument("config_class", help="path to config file for classifier model")
    parser.add_argument("checkpoint_class", help="path to checkpoint file for classifier model")
    parser.add_argument("--calgroup", "-c", type=str, help="calibration group override")
    parser.add_argument("--output", "-o", type=str, help="path to output hdf5 file")
    parser.add_argument("--scale_factor_z", "-sz", type=float, help="scale factor used in normalization before passing to z model")
    parser.add_argument("--scale_factor_class", "-sc", type=float, help="scale factor used in normalization before passing to class model")
    parser.add_argument("--cpu", "-cpu", action="store_true", help="map tensor device storage to cpu")
    parser.add_argument("--num_threads", "-nt", type=int, help="number of threads to use")
    parser.add_argument("--buffer_size", "-b", type=int, help="number of rows to store in memory before writing to disk", default=1024*16)
    parser.add_argument("--read_size", "-r", type=int, help="number of rows per read", default=2048)
    args = parser.parse_args()
    input_path = check_path(args.input_path)
    config_z = check_path(args.config_z)
    checkpoint_z = check_path(args.checkpoint_z)
    config_class = check_path(args.config_class)
    checkpoint_class = check_path(args.checkpoint_class)
    output = input_path[:input_path.rfind("_")] + "_Phys.h5"
    print("Writing phys pulse output to {}".format(output))
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
    if args.buffer_size:
        pw_args["n_buffer_rows"] = args.buffer_size
    if args.read_size:
        pw_args["n_rows_per_read"] = args.read_size
    if args.calgroup:
        pw_args["calgroup"] = args.calgroup
    else:
        pw_args["calgroup"] = os.path.basename(input_path[:input_path.rfind("_")])
    print("using calibration group {}".format(pw_args["calgroup"]))
    if args.scale_factor_z:
        pw_args["scale_factor_z"] = args.scale_factor_z
    if args.scale_factor_class:
        pw_args["scale_factor_class"] = args.scale_factor_class
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    PW = ZAndClassWriter(output, input_path, config_z, checkpoint_z, config_class, checkpoint_class, **pw_args)
    print("Writing predictions")
    PW.write_predictions()
    runtime = time.time() - start_time
    print("Success")
    print("Writing XML metadata")
    PW.write_XML(runtime)
    print("Success")


if __name__ == "__main__":
    main()
