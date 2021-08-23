import argparse
from os.path import basename, dirname, realpath
import sys
import torch
from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger
import spconv

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.util import get_config, get_tb_logdir_version, ModuleUtility


def plot_weights(model, logger):
    # extracting the model features at the particular layer number
    print(model)
    for layer_num in range(len(model)):
        layer = model[layer_num]
        # checking whether the layer is convolution layer or not
        if isinstance(layer, spconv.SparseConv2d):
            # getting the weight tensor data
            weight_tensor = model[layer_num].weight.data
            print("conv2d layer - [{0} - {1}] {2}".format(model[layer_num].in_channels, model[layer_num].out_channels, model[layer_num].kernel_size))
            print(weight_tensor.shape)
        else:
            print(layer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    parser.add_argument("checkpoint", help="path to checkpoint file")
    parser.add_argument("--num_threads", "-nt", type=int, help="number of threads to use")
    args = parser.parse_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    config = get_config(args.config)
    log_folder = dirname(args.config)
    p = Path(log_folder)
    cp = p.glob('*.tfevents.*')
    tb_logger_args = {}
    logger = None
    if cp:
        for ckpt in cp:
            print("Using existing log file {}".format(ckpt))
            vnum = get_tb_logdir_version(str(ckpt))
            logger = TensorBoardLogger(dirname(dirname(log_folder)), name=basename(dirname(log_folder)), version=vnum, **tb_logger_args)
            break
    else:
        logger = TensorBoardLogger(log_folder, name=config.run_config.exp_name, **tb_logger_args)
        print("Creating new log file in directory {}".format(logger.log_dir))
    modules = ModuleUtility(config.run_config.imports)
    model = modules.retrieve_class(config.run_config.run_class).load_from_checkpoint(args.checkpoint, config=config).model.model.network
    plot_weights(model, logger)


if __name__ == "__main__":
    main()