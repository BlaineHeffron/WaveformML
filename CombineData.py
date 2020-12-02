from src.datasets.PulseDataset import *
from src.utils.util import unique_path_combine, DictionaryUtility, check_config
from os.path import isdir, join, abspath, normpath, expanduser
import argparse

TYPES = ['2d', '3d', 'pmt', 'det']
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

def main():
    parser = argparse.ArgumentParser(description='Shuffle data from directories.')
    parser.add_argument('dirs', type=str, nargs='+',
                        help='directories to combine the file')
    parser.add_argument('--outdir', '-o', type=str,
                        help='directory to place shuffled files. If not specified, defaults to ./data/<combined '
                             'directory name>')
    parser.add_argument('--type', '-t', type=str,
                        help='Type of file. Either 3d, 2d, det or pmt. Defaults to 2d')
    parser.add_argument('--config', '-c', type=str,
                        help='Pass config file to override chunk_size and shuffled_size.')
    args = parser.parse_args()
    type = '2d'
    args.dirs = [normpath(abspath(expanduser(p))) for p in args.dirs]
    for d in args.dirs:
        if not isdir(d):
            raise IOError("Invalid directory {}".format(d))
    outdir = join("./data/", unique_path_combine(args.dirs))
    if args.type:
        if args.type not in TYPES:
            raise IOError("type must be one of {0}".format(TYPES))
        else:
            type = args.type
    if args.outdir:
        if not isdir(normpath(abspath(expanduser(args.outdir)))):
            print("Warning: specified output directory {} does not exist, attempting to create it")
            os.makedirs(normpath(abspath(expanduser(args.outdir))), exist_ok=True)
        outdir = normpath(abspath(expanduser(args.outdir)))
    else:
        outdir = normpath(abspath(expanduser(outdir)))
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
    setattr(config.dataset_config, "data_prep", "shuffle")
    if not hasattr(config.dataset_config, "shuffled_size"):
        setattr(config.dataset_config, "shuffled_size", 16384)
    if not hasattr(config.dataset_config, "chunk_size"):
        setattr(config.dataset_config, "chunk_size", 1024)
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
    print("Writing combined files to {}".format(outdir))
    d.write_shuffled()


if __name__ == '__main__':
    main()
