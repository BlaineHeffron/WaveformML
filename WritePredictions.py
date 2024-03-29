import argparse
from os.path import expanduser, isdir, join
from ntpath import basename
import torch

from src.datasets.PredictionWriter import *
from src.utils.util import check_path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="path to input hdf5 file")
    parser.add_argument("config", help="path to config file for model")
    parser.add_argument("checkpoint", help="path to checkpoint file for model")
    parser.add_argument("--writer", "-w", type=str, help="choose writer. 'z' for Z prediction writer, 'irn' for IRN prediction writer, 'irnim' for INRIM prediction writer")
    parser.add_argument("--output", "-o", type=str, help="path to output hdf5 file")
    parser.add_argument("--calgroup", "-c", type=str, help="calibration group to use for E prediction / normalization when passing WaveformPairCal")
    parser.add_argument("--scale_factor", "-s", type=float, help="scale factor used in normalization before passing to model (when passing WaveformPairCal")
    parser.add_argument("--datatype", "-d", type=str, help="name of datatype to use to override config file (ie use this when skipping normalization stage, pass 'WaveformPairCal')")
    parser.add_argument("--cpu", "-cpu", action="store_true", help="map tensor device storage to cpu")
    parser.add_argument("--num_threads", "-nt", type=int, help="number of threads to use")
    parser.add_argument("--buffer_size", "-b", type=int, help="number of rows to store in memory before writing to disk", default=1024*16)
    parser.add_argument("--read_size", "-r", type=int, help="number of rows per read", default=2048)

    args = parser.parse_args()
    input_path = check_path(args.input_path)
    config = check_path(args.config)
    checkpoint = check_path(args.checkpoint)
    if args.datatype == "PhysPulse":
        output = input_path[:input_path.rfind("_")] + "_Phys.h5"
        print("Writing phys pulse output to {}".format(output))
    else:
        output = input_path[0:-3] + "ModelOut.h5"
        print("Writing output to {0}".format(output))
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
        pw_args["calgroup"] = basename(input_path[:input_path.rfind("_")])
    if args.scale_factor:
        pw_args["scale_factor"] = args.scale_factor
    if args.datatype:
        pw_args["datatype"] = args.datatype
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    if not args.writer:
        print("no writer selected, using ZPredictionWriter")
        PW = ZPredictionWriter(output, input_path, config, checkpoint, **pw_args)
    elif args.writer == "z":
        PW = ZPredictionWriter(output, input_path, config, checkpoint, **pw_args)
    elif args.writer == "irn":
        PW = IRNPredictionWriter(output, input_path, config, checkpoint, **pw_args)
    elif args.writer == "irnim":
        PW = IRNIMPredictionWriter(output, input_path, config, checkpoint, **pw_args)
    else:
        raise IOError("{} not a valid choice for writer. ".format(args.writer))
    print("Writing predictions")
    PW.write_predictions()
    runtime = time.time() - start_time
    print("Success")
    print("Writing XML metadata")
    PW.write_XML(runtime)
    print("Success")


if __name__ == "__main__":
    main()