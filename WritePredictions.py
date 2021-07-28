import argparse
from os.path import expanduser, isdir, isfile, join
from ntpath import basename

from src.datasets.PredictionWriter import *
from src.utils.util import check_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="path to input hdf5 file")
    parser.add_argument("config", help="path to config file for model")
    parser.add_argument("checkpoint", help="path to checkpoint file for model")
    parser.add_argument("--output", "-o", type=str, help="path to output hdf5 file")

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
    PW = ZPredictionWriter(output, input_path, config, checkpoint)
    print("Writing predictions")
    PW.write_predictions()
    print("Success")
    print("Writing XML metadata")
    PW.write_XML()


if __name__ == "__main__":
    main()