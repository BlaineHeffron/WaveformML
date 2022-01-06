import sys
from os.path import dirname, realpath
import argparse

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.SQLUtils import CalibrationDB
from src.evaluation.Calibrator import Calibrator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cal1", help="cal path 1")
    parser.add_argument("cal2", help="cal path 2")
    parser.add_argument("calname", help="cal path 2")
    args = parser.parse_args()
    cal1 = CalibrationDB(args.cal1, args.calname)
    cal2 = CalibrationDB(args.cal2, args.calname)
    cal1 = Calibrator(cal1)
    cal2 = Calibrator(cal2)
    max_diff = 0
    max_det = 0
    for i in range(14):
        for j in range(11):
            for k in range(2):
                diff = (cal1.gains[i,j,k] - cal2.gains[i,j,k]) / cal1.gains[i,j,k]
                if(abs(diff) > max_diff):
                    max_det = 2*(14*j + i) + k
                    max_diff = abs(diff)
                print("det {0} gain diff {1}".format((2*(14*j + i) + k), diff))
    print("max diff is det {0} diff {1}".format(max_det, max_diff))
    #print((cal1.gains - cal2.gains) / cal1.gains)


if __name__=="__main__":
    main()