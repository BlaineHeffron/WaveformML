
from os.path import dirname, realpath
import os
import sys
from CompareCalibrationCurves import WFCompare
from src.utils.util import json_load

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.SQLUtils import WFParamsDB
import argparse
from CompareCalibrationCurves import plot_and_print_cal_curves


class WFParamEvaluator(WFParamsDB):
    def __init__(self, dbnm, calname):
        super(WFParamEvaluator, self).__init__(dbnm)
        self.calname = calname
        self.wfcompare = None

    def eval_wf_params(self):
        result = self.retrieve_simnames_for_eval(self.calname)
        if result is not None and len(result) > 0:
            for row in result:
                id, name = row
                print("name : {}".format(name))
                if self.wfcompare is None:
                    self.wfcompare = WFCompare(self.calname, name, "")
                else:
                    self.wfcompare.swap_sim(name)
                for seg in range(14*11):
                    params = self.wfcompare.compare_seg(seg)
                    self.insert_eval_for_seg(self.calname, seg, id, params)
                self.commit()
        else:
            print("Warning: no sim names for calname {}".format(self.calname))

    def best_fits_per_seg(self, calname, printed_params=None, min=None, max=None, limit=None):
        if printed_params is None:
            printed_params = ["PE_per_MeV", "lambda", "PMT_sigma_t", "n", "zoff"]
        print("best fits found for calname {}".format(calname))
        if limit is None:
            limit = 1
        printstr = " | ".join(printed_params)
        print("| seg | sim calname | normed diff | at0 | at1 | t0 | t1 | psd0 | psd1 | {0} |".format(printstr))
        for seg in range(14*11):
            result = self.query_smallest_diffs(calname, seg, printed_params, limit, min=min, max=max)
            if result:
                for row in result:
                    row = ["{:.3f}".format(r) if isinstance(r, float) else str(r) for r in row]
                    print("| {} |".format(" | ".join(row)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calname", help="calibration name")
    parser.add_argument("outdir", help="directory to plot sample comparison")
    parser.add_argument("example_seg", type=int, default=0, help="example seg to plot best fit of (default 0)")
    parser.add_argument("--config", "-c", help="use config file used for GenWFParamConfig to set which params to display")
    parser.add_argument("--limit", type=int, help="number of cals to display per segment (default is top 1)", default=1)
    parser.add_argument("--mincal","-l", type=int, help="minimum cal number to select from ")
    parser.add_argument("--maxcal","-m", type=int, help="maximum cal number to select from ")
    args = parser.parse_args()
    wfpe = WFParamEvaluator(os.environ["WF_PARAMS_DB"], args.calname)
    wfpe.eval_wf_params()
    argdict = {}
    diffs_dict = {}
    if args.mincal:
        argdict["min"] = args.mincal
        diffs_dict["min"] = args.mincal
    if args.maxcal:
        argdict["max"] = args.maxcal
        diffs_dict["max"] = args.maxcal
    if args.config:
        json = json_load(args.config)
        if "param_ranges" in json.keys():
            argdict["printed_params"] = [p for p in json["param_ranges"].keys()]
            diffs_dict["params"] = [p for p in json["param_ranges"].keys()]
    wfpe.best_fits_per_seg(args.calname, limit=args.limit, **argdict)
    seg = args.example_seg
    result = wfpe.query_smallest_diffs(args.calname, seg, limit=1, **diffs_dict)
    if result is not None:
        row = result[0]
        name = row[1]
        plot_and_print_cal_curves(args.calname, name, seg, args.outdir)


if __name__ == "__main__":
    main()