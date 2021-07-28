import argparse
from os.path import expanduser

from src.datasets.PredictionWriter import *
from src.utils.util import check_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="path to input hdf5 file")
    parser.add_argument("output_path", help="path to output hdf5 file")
    parser.add_argument("config", help="path to config file for model")
    parser.add_argument("checkpoint", help="path to checkpoint file for model")

    args = parser.parse_args()
    input_path = check_path(args.input_path)
    config = check_path(args.config)
    checkpoint = check_path(args.checkpoint)
    print("Setting up output")
    PW = ZPredictionWriter(expanduser(args.output_path), input_path, config, checkpoint)
    print("Writing predictions")
    PW.write_predictions()
    print("Success")
    print("Writing XML metadata")
    PW.write_XML()


if __name__ == "__main__":
    main()