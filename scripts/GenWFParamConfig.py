import argparse
from copy import copy
from os.path import join, dirname, realpath
import os
import sys
from typing import Dict

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.util import json_load
from src.utils.SQLUtils import WFParamsDB

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

class WaveformParamSet:
    def __init__(self):
        self.PE_per_MeV = 1200
        self.gain = -5000
        self.PMT_sigma_t = 3
        self.lamb = 1775
        self.n = 1.6
        self.zoff = 1.0
        self.x_crit = 0
        self.lambda_s = 0
        self.eta_bar = 0
        self.PMT_decay_proportion_1 = 0.6
        self.PMT_decay_proportion_2 = 0.4
        self.PMT_decay_tau_1 = 0.5
        self.PMT_decay_tau_2 = 16
        self.PSD_response_1_p1 = 0.7
        self.PSD_response_1_p2 = 0.28
        self.PSD_response_1_p3 = 0.02
        self.PSD_response_1_tau1 = 3.16
        self.PSD_response_1_tau2 = 32.3
        self.PSD_response_1_tau3 = 270
        self.PSD_response_2_p1 = 0.3
        self.PSD_response_2_p2 = 0.65
        self.PSD_response_2_p3 = 0.05
        self.PSD_response_2_tau1 = 3.16
        self.PSD_response_2_tau2 = 32.3
        self.PSD_response_2_tau3 = 270
        self.name = None
        self.param_ranges = {}
        self.num_points = 1

    def set_ranges(self, range_dict):
        self.param_ranges.update(range_dict)
        for range in self.param_ranges.values():
            range[0] = float(range[0])
            range[1] = float(range[1])

    def set_parameter(self, name, value):
        if name == "lambda":
            self.lamb = value
        else:
            setattr(self, name, value)

    def get_dict(self):
        this_dict = copy(self.__dict__)
        del this_dict["param_ranges"]
        del this_dict["num_points"]
        if "lamb" in this_dict.keys():
            this_dict["lambda"] = this_dict["lamb"]
            del this_dict["lamb"]
        return this_dict

    def gen_parameters(self, n):
        i = 0
        num_base = numberToBase(n, self.num_points)
        for param, par_range in self.param_ranges.items():
            inc = (par_range[1] - par_range[0]) / (self.num_points - 1)
            if len(num_base) < (i+1):
                val = par_range[0]
            else:
                val = inc*num_base[i] + par_range[0]
            self.set_parameter(param, val)
            i += 1

    def load_config(self, config_dict):
        if "param_ranges" not in config_dict.keys():
            raise IOError("param_ranges must be in WFParam config")
        if "num_points" not in config_dict.keys():
            raise IOError("num_points must be in WFParam config")
        self.set_ranges(config_dict["param_ranges"])
        if "defaults" in config_dict.keys():
            self.set_defaults(config_dict["defaults"])
        self.num_points = int(config_dict["num_points"])

    def set_defaults(self, defaults: Dict):
        for param, val in defaults.items():
            self.set_parameter(param, val)

    def total_points(self):
        return self.num_points**(len(self.param_ranges.keys()))

    def gen_configs(self, db: WFParamsDB, path, template):
        first_name = db.get_unique_name()
        first_num = int(first_name[7:])
        for i in range(self.total_points()):
            self.gen_parameters(i)
            self.name = "WaveCal{}".format(first_num + i)
            config_name = self.name + ".cfg"
            mydict = self.get_dict()
            with open(join(path, config_name), 'w') as configf:
                configf.write(template%mydict)
            db.insert_set(mydict)
        db.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file for parameter ranges to vary over (json)")
    parser.add_argument("path", help="path to generate config files for P2x")
    parser.add_argument("--db", "-d", help="optional database to store in (defaults to $WF_PARAMS_DB")
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    template = open(join(join(dir_path,"templates"),"WFParams.cfg"), "r").read()
    mypath = realpath(os.path.expanduser(args.path))
    config = json_load(realpath(os.path.expanduser(args.config)))
    if args.db:
        pardb = WFParamsDB(args.db)
    else:
        if "WF_PARAMS_DB" not in os.environ.keys():
            raise IOError("must either pass --db option to specify waveform params db or have $WF_PARAMS_DB environment variable set")
        pardb = WFParamsDB(os.environ["WF_PARAMS_DB"])
    param_set = WaveformParamSet()
    param_set.load_config(config)
    param_set.gen_configs(pardb, mypath, template)



if __name__=="__main__":
    main()



