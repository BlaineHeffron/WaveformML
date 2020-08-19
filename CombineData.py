from main import check_config
from src.datasets.PulseDataset import *
from src.utils.util import unique_path_combine, DictionaryUtility
from os.path import isdir, join, abspath, normpath, expanduser
import argparse

TYPES = ['2d', '3d']
DEFAULT_CONFIG = {
    "dataset_config": {
        "data_prep": "shuffle",
        "chunk_size": 1024,
        "shuffled_filesize": 16384,
        "dataset_params": {
            "data_cache_size": 1
        }
    }
}


def main():
    parser = argparse.ArgumentParser(description='Shuffle data from directories.')
    parser.add_argument('dirs', type=str, nargs='+',
                        help='directories to combine the file')
    parser.add_argument('--outdir', '-o', type=str,
                        help='directory to place shuffled files. If not specified, defaults to ./data/<combined '
                             'directory name>')
    parser.add_argument('--type', '-t', type=str,
                        help='Type of file. Either 3d or 2d. Defaults to 2d')
    parser.add_argument('--config', '-c', type=str,
                        help='Pass config file to override chunk_size and shuffled_filesize.')
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
    if args.config:
        config = args.config
        config = check_config(config)
        with open(config) as json_data_file:
            config = json.load(json_data_file)
    config = DictionaryUtility.to_object(config)
    setattr(config.dataset_config, "data_prep", "shuffle")
    if not hasattr(config.dataset_config, "shuffled_filesize"):
        setattr(config.dataset_config, "shuffled_filesize", 16384)
    if not hasattr(config.dataset_config, "chunk_size"):
        setattr(config.dataset_config, "chunk_size", 1024)
    setattr(config.dataset_config, "base_path", "")
    setattr(config.dataset_config, "paths", args.dirs)
    if type == '2d':
        d = PulseDataset2D(config, "train", 1000000000, "cpu0", model_dir="./model", data_dir=outdir,
                           dataset_dir=outdir)
    elif type == '3d':
        d = PulseDataset2D(config, "train", 1000000000, "cpu0", model_dir="./model", data_dir=outdir,
                           dataset_dir=outdir)
    else:
        raise IOError("Unknown dataset type {}".format(type))
    print("Writing combined files to {}".format(outdir))
    d.write_shuffled()


if __name__ == '__main__':
    main()