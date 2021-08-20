import argparse
from pathlib import Path
from os.path import join, exists, basename, dirname, realpath
import sys
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.TensorBoardUtils import TBHelper
import numpy as np
from src.utils.PlotUtils import ScatterPlt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="directory of occlude_# folders")
    parser.add_argument("n", help="number of feature occluded")
    args = parser.parse_args()
    num_ind = int(args.n)
    results = np.zeros((num_ind,))
    metric_name = "test_loss"
    p = Path(args.dir)
    tbh = TBHelper()
    plot_path = join(args.dir, "occlude_results_{}.png".format(metric_name))
    plot_path_rel = join(args.dir, "occlude_results_{}_relative.png".format(metric_name))
    for dir in p.glob("occlude_*"):
        if not dir.is_dir:
            continue
        try:
            occlude_ind = int(dir.name.split("_")[-1])
        except ValueError as e:
            print("directory {} bad directory, skipping".format(str(dir.resolve())))
            continue
        if occlude_ind >= num_ind:
            continue
        thisdir = str(dir.resolve())
        occlude_dir = Path(thisdir)
        logfiles = occlude_dir.glob("*events.out.tfevents.*")
        best_loss = 100000000
        for f in logfiles:
            tbh.set_file(str(f.resolve()))
            file_loss = tbh.get_best_value(metric_name)
            if file_loss is not None:
                if best_loss > file_loss:
                    best_loss = file_loss
        results[occlude_ind] = best_loss
        print("{0} for ind {1} is {2}".format(metric_name, occlude_ind, best_loss))
    print("outputting results to plot {}".format(plot_path))
    ScatterPlt([i for i in range(num_ind)], results, "feature index occluded", metric_name, plot_path, title=metric_name + " for occluded features")
    results_rel = results - np.min(results)
    ScatterPlt([i for i in range(num_ind)], results_rel, "feature index occluded", "additional " + metric_name, plot_path_rel, title=metric_name + " difference for occluded features")



if __name__ == "__main__":
    main()
