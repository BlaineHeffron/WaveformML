from CompareCalibrationCurves import SegCompare
import argparse
from math import floor
from os.path import join, exists, basename, dirname, realpath
import os
import sys
import numpy as np
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.SQLUtils import CalibrationDB, CalCurve
from src.evaluation.Calibrator import Calibrator
import csv


def write_csv(name, data):
    with open(name, 'w') as csvfile:
        writer = csv.writer(csvfile,
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("simcalname", help="calibration name of simulation (assumes seg = 0)")
    parser.add_argument("realcalname", help="calibration name from real data to compare")
    parser.add_argument("outdir", help="directory to save tables")
    args = parser.parse_args()
    calsim = CalibrationDB(os.environ["PROSPECT_CALDB"], args.simcalname)
    calreal = CalibrationDB(os.environ["PROSPECT_CALDB"], args.realcalname)
    calsim = Calibrator(calsim)
    calreal = Calibrator(calreal)
    dir = os.path.normpath(os.path.expanduser(args.outdir))
    diffs = [["seg", "total normed diff", "normed attenuation diff", "normed time diff", "absolute attenuation diff 0", "absolute attenuation diff 1", "absolute time diff 0", "absolute time diff 1"]]
    for i in range(14*11):
        AttenCompare = SegCompare(calsim.atten_curves, calreal.atten_curves, 0, dir, args.simcalname, args.realcalname, "attenuation", seg2=i)
        TimeCompare = SegCompare(calsim.time_curves, calreal.time_curves, 0, dir, args.simcalname, args.realcalname, "timing", seg2=i)
        difT = TimeCompare.normed_diff()
        difA = AttenCompare.normed_diff()
        diffs.append([i, difT + difA, difA, difT, AttenCompare.CC0.abs_diff(), AttenCompare.CC1.abs_diff(), TimeCompare.CC0.abs_diff(), TimeCompare.CC1.abs_diff()])

    write_csv(os.path.join(dir, "cal_curves_diffs.txt"), diffs)
    diff_arr = np.array(diffs[1:])
    n = len(diff_arr)
    ave_diff = sum(diff_arr[:,1]) / n
    sdev_diff = 0
    for i in range(diff_arr.shape[0]):
        sdev_diff += (diff_arr[i, 1] - ave_diff)**2
    var_diff = (sdev_diff / (n - 1))**0.5
    print("average difference : {0}, std dev: {1}".format(ave_diff, var_diff))
    print("max difference: {0}, min difference: {1}".format(np.max(diff_arr[:,1]), np.min(diff_arr[:,1])))


if __name__ == "__main__":
    main()