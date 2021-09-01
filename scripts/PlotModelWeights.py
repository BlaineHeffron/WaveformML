import argparse
from os.path import basename, dirname, realpath
import sys
import torch
from pathlib import Path
from numpy import arange

from pytorch_lightning.loggers import TensorBoardLogger
import spconv
import matplotlib.pyplot as plt

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.util import get_config, ModuleUtility
from src.utils.PlotUtils import plot_hist2d

def plot_2d_tensor(tensor, name, fname, logger : TensorBoardLogger):
    xaxis = arange(-0.5, tensor.shape[0] - 0.49, 1)
    yaxis = arange(-0.5, tensor.shape[1] - 0.49, 1)
    fig = plot_hist2d(xaxis, yaxis, tensor.detach().cpu().numpy(), name, "channel in dimension", "channel out dimension", "weight value", norm_to_bin_width=False, logz=False)
    logger.experiment.add_figure("model/{}".format(fname), fig)




def plot_weights(model, logger):
    # extracting the model features at the particular layer number
    print(model)
    for layer_num in range(len(model)):
        layer = model[layer_num]
        # checking whether the layer is convolution layer or not
        if isinstance(layer, spconv.SparseConv2d):
            # getting the weight tensor data
            weight_tensor = model[layer_num].weight.data
            if weight_tensor.shape[0] == 1 and weight_tensor.shape[1] == 1:
                plot_2d_tensor(weight_tensor[0,0], "layer {}".format(layer_num), "layer_{}.png".format(layer_num), logger)
            print("conv2d layer - [{0} - {1}] {2}".format(model[layer_num].in_channels, model[layer_num].out_channels, model[layer_num].kernel_size))
        else:
            print(layer)

def plot_waveform_evolution(model, layer_nums, wf):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook
    for layer_num in layer_nums:
        model.layer[layer_num].register_forward_hook(get_activation('layer_{}'.format(layer_num)))
    coo = torch.tensor([8, 2, 0]).unsqueeze_(0)
    wf.unsqueeze_(0)
    output = model([wf, coo])
    for layer_num in layer_nums:
        act = activation['layer_{}'.format(layer_num)].squeeze()
        fig, axarr = plt.subplots(act.size(0))
        for idx in range(act.size(0)):
            axarr[idx].imshow(act[idx])

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
            logger = TensorBoardLogger(dirname(dirname(log_folder)), name=basename(dirname(log_folder)), version=basename(log_folder), **tb_logger_args)
            break
    else:
        logger = TensorBoardLogger(log_folder, name=config.run_config.exp_name, **tb_logger_args)
        print("Creating new log file in directory {}".format(logger.log_dir))
    modules = ModuleUtility(config.run_config.imports)
    model = modules.retrieve_class(config.run_config.run_class).load_from_checkpoint(args.checkpoint, config=config).model.model.network
    plot_weights(model, logger)
    logger.finalize("done")
    logger.save()
    logger.close()


if __name__ == "__main__":
    main()