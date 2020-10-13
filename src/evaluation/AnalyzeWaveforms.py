import json
from os.path import normpath, abspath, expanduser, isdir, join

from src.datasets.PulseDataset import *
from src.utils.SparseUtils import average_pulse, find_matches, metric_accumulate_2d, metric_accumulate_1d, \
    get_typed_list, weighted_average_quantities
import argparse
from numpy import int32, zeros, sum, sqrt, float32

from src.utils.util import unique_path_combine, check_config, DictionaryUtility

DEFAULT_CONFIG = {
    "dataset_config": {
        "data_prep": "shuffle",
        "chunk_size": 1024,
        "shuffled_size": 16384,
        "dataset_params": {
            "data_cache_size": 1,
            "label_file_pattern": "*WaveformPairSimLabel.h5"
        }
    }
}

CONFIG_DIR = "./config"

class WaveformAccumulator:
    def __init__(self, s):
        self.s = s
        self.clear()

    def add(self,wfs):
        self.wf += sum(wfs,axis=0)
        self.total += 1

    def compute(self):
        if self.total > 0:
            return self.wf/self.total, sqrt(self.wf)
        else:
            print("Warning: nothing to compute!")
            return 0, 0

    def clear(self):
        self.wf = zeros((self.s,),dtype=int32)
        self.total = 0



def main():
    parser = argparse.ArgumentParser(description='Pass in directories of data  for histogramming.')
    parser.add_argument('dirs', type=str, nargs='+',
                        help='directories to combine the file')
    parser.add_argument('--outdir', '-o', type=str,
                        help='directory to place shuffled files. If not specified, defaults to ./data/<combined '
                             'directory name>')
    parser.add_argument('--type', '-t', type=str,
                        help='Type of file. Either 3d, 2d, det or pmt. Defaults to 2d')
    args = parser.parse_args()
    type = '2d'
    args.dirs = [normpath(abspath(expanduser(p))) for p in args.dirs]
    for d in args.dirs:
        if not isdir(d):
            raise IOError("Invalid directory {}".format(d))
    outdir = join("./analysis/", unique_path_combine(args.dirs))
    if args.type:
        type = args.type
    config = DEFAULT_CONFIG
    if type == 'pmt':
        config['dataset_config']['dataset_params']['label_file_pattern'] = "*PMTCoordSimLabel.h5"
    elif type == '3d':
        config['dataset_config']['dataset_params']['label_file_pattern'] = "*Waveform3DPairSimLabel.h5"
    elif type == 'det':
        config['dataset_config']['dataset_params']['label_file_pattern'] = "*DetCoordSimLabel.h5"
    if args.config:
        config = args.config
        config = check_config(config, CONFIG_DIR)
        with open(config) as json_data_file:
            config = json.load(json_data_file)
    config = DictionaryUtility.to_object(config)
    setattr(config.dataset_config, "base_path", "")
    setattr(config.dataset_config, "paths", args.dirs)
    if type == '2d':
        d = PulseDataset2D(config, "train", 1000000000, "cpu0", model_dir="./model", data_dir=outdir,
                           dataset_dir=outdir, **DictionaryUtility.to_dict(config.dataset_config.dataset_params))
    elif type == '3d':
        d = PulseDataset2D(config, "train", 1000000000, "cpu0", model_dir="./model", data_dir=outdir,
                           dataset_dir=outdir, **DictionaryUtility.to_dict(config.dataset_config.dataset_params))
    elif type == 'pmt':
        d = PulseDatasetPMT(config, "train", 1000000000, "cpu0", model_dir="./model", data_dir=outdir,
                            dataset_dir=outdir, **DictionaryUtility.to_dict(config.dataset_config.dataset_params))
    elif type == 'det':
        d = PulseDatasetDet(config, "train", 1000000000, "cpu0", model_dir="./model", data_dir=outdir,
                            dataset_dir=outdir, **DictionaryUtility.to_dict(config.dataset_config.dataset_params))
    else:
        raise IOError("Unknown dataset type {}".format(type))
    class_names = config.system_config.type_names
    n_samples = config.system_config.n_samples
    accumulators = []
    for cname in class_names:
        accumulators.append(WaveformAccumulator(n_samples))
    for batch in d:
        (c, f), labels = batch
        c, f, labels = c.detach().cpu().numpy(), f.detach().cpu().numpy(), labels.detach().cpu().numpy()
        avg_coo, summed_pulses, multiplicity, psdl, psdr = average_pulse(c, f,
                                                                         zeros((labels.shape[0], 2)),
                                                                         zeros((labels.shape[0], f.shape[1],),
                                                                               dtype=float32),
                                                                         zeros((labels.shape[0],), dtype=int32),
                                                                         zeros((labels.shape[0],),
                                                                               dtype=float32),
                                                                         zeros((labels.shape[0],),
                                                                               dtype=float32))





if __name__ == "__main__":
    main()