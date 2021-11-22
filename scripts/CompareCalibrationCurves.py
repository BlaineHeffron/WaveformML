import argparse
from math import floor
from os.path import join, exists, basename, dirname, realpath
import os
import sys
import numpy as np
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.SQLUtils import CalibrationDB, CalCurve
from src.evaluation.Calibrator import Calibrator

from src.utils.PlotUtils import MultiLinePlot
from src.utils.SparseUtils import lin_interp
import matplotlib.pyplot as plt

def rescale_psd(curve: CalCurve):
    n = int(len(curve.ys))
    val = curve.ys[int((n - 1) / 2)]
    curve.ys = [y/val for y in curve.ys]


def convert_psd(curve: CalCurve, time_pos_curves):
    if curve.xs[0] > -40:
        curve.xs.reverse()
        curve.xs = [lin_interp(time_pos_curves, dt) for dt in curve.xs]
        curve.ys.reverse()

def plot_and_print_cal_curves(name1, name2, seg, outdir):
    cal1 = CalibrationDB(os.environ["PROSPECT_CALDB"], name1)
    cal2 = CalibrationDB(os.environ["PROSPECT_CALDB"], name2)
    cal1 = Calibrator(cal1)
    cal2 = Calibrator(cal2)
    dir = os.path.normpath(os.path.expanduser(outdir))
    x = seg % 14
    y = floor(seg / 14)
    #labels = ["{0} seg {1} pmt 0".format(name1, seg), "{0} seg {1} pmt 1".format(name1, seg), "{0} seg {1} pmt 0".format(name2, seg), "{0} seg {1} pmt 1".format(name2, seg)]
    rescale_psd(cal1.psd_curves[seg*2])
    rescale_psd(cal1.psd_curves[seg*2 + 1])
    rescale_psd(cal2.psd_curves[seg*2])
    rescale_psd(cal2.psd_curves[seg*2 + 1])
    convert_psd(cal1.psd_curves[seg*2], cal1.time_pos_curves[x, y])
    convert_psd(cal1.psd_curves[seg*2 + 1], cal1.time_pos_curves[x, y])
    convert_psd(cal2.psd_curves[seg*2], cal2.time_pos_curves[x, y])
    convert_psd(cal2.psd_curves[seg*2 + 1], cal2.time_pos_curves[x, y])
    PSDCompare = SegCompare(cal1.psd_curves, cal2.psd_curves, seg, dir, name1, name2, "PSD")
    AttenCompare = SegCompare(cal1.atten_curves, cal2.atten_curves, seg, dir, name1, name2, "attenuation")
    TimeCompare = SegCompare(cal1.time_curves, cal2.time_curves, seg, dir, name1, name2, "timing")
    PSDCompare.plot()
    AttenCompare.plot()
    TimeCompare.plot()
    PSDCompare.print_diff()
    AttenCompare.print_diff()
    TimeCompare.print_diff()
    print("| PSD norm diff 0 | PSD norm diff 1 | Atten norm diff 0 | Atten norm diff 1 | time norm diff 0 | time norm diff 1 |")
    print("| {0} | {1} | {2} | {3} | {4} | {5} |".format(PSDCompare.CC0.normed_abs_diff(), PSDCompare.CC1.normed_abs_diff(),
                                                         AttenCompare.CC0.normed_abs_diff(), AttenCompare.CC1.normed_abs_diff(),
                                                         TimeCompare.CC0.normed_abs_diff(), TimeCompare.CC1.normed_abs_diff()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name1", help="calibration name 1")
    parser.add_argument("name2", help="calibration name 2")
    parser.add_argument("seg", help="segment for comparison", type=int)
    parser.add_argument("outdir", help="directory to save plots")
    args = parser.parse_args()
    plot_and_print_cal_curves(args.name1,args.name2,args.seg,args.outdir)

class WFCompare:
    def __init__(self, realname, simname, outdir):
        self.dir = outdir
        self.realname = realname
        self.simname = simname
        realcal = CalibrationDB(os.environ["PROSPECT_CALDB"], realname)
        simcal = CalibrationDB(os.environ["PROSPECT_CALDB"], simname)
        self.realcal = Calibrator(realcal)
        self.simcal = Calibrator(simcal)
        self.psd_rescale = set()
        self.rescal_sim_psd()

    def rescal_sim_psd(self):
        rescale_psd(self.simcal.psd_curves[0])
        rescale_psd(self.simcal.psd_curves[1])
        convert_psd(self.simcal.psd_curves[0], self.simcal.time_pos_curves[0,0])
        convert_psd(self.simcal.psd_curves[1], self.simcal.time_pos_curves[0,0])

    def swap_sim(self, name):
        simcal = CalibrationDB(os.environ["PROSPECT_CALDB"], name)
        self.simcal = Calibrator(simcal)
        self.rescal_sim_psd()


    def compare_seg(self, seg):
        x = seg % 14
        y = floor(seg / 14)
        if seg not in self.psd_rescale:
            self.psd_rescale.add(seg)
            self.psd_rescale.add(seg*2 + 1)
            rescale_psd(self.realcal.psd_curves[seg * 2])
            rescale_psd(self.realcal.psd_curves[seg * 2 + 1])
            convert_psd(self.realcal.psd_curves[seg * 2], self.realcal.time_pos_curves[x, y])
            convert_psd(self.realcal.psd_curves[seg *2 + 1], self.realcal.time_pos_curves[x, y])
        PSDCompare = SegCompare(self.realcal.psd_curves, self.simcal.psd_curves, seg, self.dir, self.realname, self.simname, "PSD", seg2=0)
        AttenCompare = SegCompare(self.realcal.atten_curves, self.simcal.atten_curves, seg, self.dir, self.realname, self.simname, "attenuation", seg2=0)
        TimeCompare = SegCompare(self.realcal.time_curves, self.simcal.time_curves, seg, self.dir, self.realname, self.simname, "timing", seg2=0)
        return *PSDCompare.normed_diff_det(), *AttenCompare.normed_diff_det(), *TimeCompare.normed_diff_det()



class SegCompare:
    def __init__(self, curves1, curves2, seg, outdir, calname0, calname1, curvename, seg2=None):
        if seg2 is None:
            self.CC0 = CurveCompare(curves1[seg * 2], curves2[seg * 2])
            self.CC1 = CurveCompare(curves1[seg*2 + 1], curves2[seg*2 + 1])
        else:
            self.CC0 = CurveCompare(curves1[seg * 2], curves2[seg2 * 2])
            self.CC1 = CurveCompare(curves1[seg * 2 + 1], curves2[seg2 * 2 + 1])
        self.outdir = outdir
        self.name = curvename
        self.seg = seg
        self.calname0 = calname0
        self.calname1 = calname1

    def plot(self):
        x0, y0 = self.CC0.common_axes()
        x1, y1 = self.CC1.common_axes()
        labels = ["cal {0} seg {1} det 0".format(self.calname0, self.seg)]
        labels += ["cal {0} seg {1} det 0".format(self.calname1, self.seg)]
        MultiLinePlot(x0, y0, labels, "position [mm]", self.name, ylog=False)
        plt.savefig(join(self.outdir, self.name + "_compare0.png"))
        labels = ["cal {0} seg {1} det 1".format(self.calname0, self.seg)]
        labels += ["cal {0} seg {1} det 1".format(self.calname1, self.seg)]
        MultiLinePlot(x1, y1, labels, "position [mm]", self.name, ylog=False)
        plt.savefig(join(self.outdir, self.name + "_compare1.png"))

    def print_diff(self):
        print("{0} absolute difference det 0: {1}".format(self.name, self.CC0.abs_diff()))
        print("{0} absolute difference det 1: {1}".format(self.name, self.CC1.abs_diff()))
        print("{0} absolute difference det 0 normed: {1}".format(self.name, self.CC0.normed_abs_diff()))
        print("{0} absolute difference det 1 normed: {1}".format(self.name, self.CC1.normed_abs_diff()))

    def normed_diff(self):
        return self.CC0.normed_abs_diff() + self.CC1.normed_abs_diff()
    
    def normed_diff_det(self):
        return self.CC0.normed_abs_diff(), self.CC1.normed_abs_diff()

class CurveCompare:
    def __init__(self, c1: CalCurve, c2: CalCurve):
        self.curve1 = c1
        self.curve2 = c2
        if abs(c1.xs[0]) < abs(c2.xs[0]):
            self.min_x = c1.xs
        else:
            self.min_x = c2.xs
        self.y1 = [self.curve1.eval(x) for x in self.min_x]
        self.y2 = [self.curve2.eval(x) for x in self.min_x]

    def common_axes(self):
        return self.min_x, [self.y1, self.y2]

    def abs_diff(self):
        return np.sum(np.abs(np.array(self.y1) - np.array(self.y2))) / len(self.y1)

    def normed_abs_diff(self):
        return self.abs_diff() * 2 * len(self.y1) / (np.sum(np.abs(self.y1)) + np.sum(np.abs(self.y2)))



if __name__ == "__main__":
    main()
