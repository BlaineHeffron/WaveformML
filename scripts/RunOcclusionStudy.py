import argparse
import subprocess
import os
import sys
from os.path import join

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.utils.util import check_path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    parser.add_argument("checkpoint", help="path to checkpoint file")
    parser.add_argument("n_features", type=int, help="number of features to occlude")
    parser.add_argument("--calgroup", "-c", help="calibration group entry in PROSPECT_CALDB", type=str)
    parser.add_argument("--onnx", "-o", action="store_true", help="set to generate onnx model instead of evaluating")
    parser.add_argument("--occlude", "-oc", type=int, default=-1, help="feature index to zero out during evaluation")
    parser.add_argument("--num_threads", "-nt", type=int, help="number of threads to use")
    parser.add_argument("--verbosity", "-v",
                        help="Set the verbosity for this run.",
                        type=int, default=0)
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    for n in range(args.n_features):
        if len(sys.argv) > 4:
            argl = ['python', join(dir_path, 'Evaluate.py'), args.config, args.checkpoint, "-oc", str(n), *sys.argv[4:]]
        else:
            argl = ['python', join(dir_path, 'Evaluate.py'), args.config, args.checkpoint, "-oc", str(n)]
        print(" ".join(argl))
        subprocess.call(argl)

if __name__ == "__main__":
    main()
