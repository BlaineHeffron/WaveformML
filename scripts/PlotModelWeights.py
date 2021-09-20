import argparse
from os.path import basename, dirname, realpath
import sys
import torch
from pathlib import Path
from numpy import arange
from torch import float32
from spconv import SparseConvTensor

from pytorch_lightning.loggers import TensorBoardLogger
import spconv
import matplotlib.pyplot as plt

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.util import get_config, ModuleUtility
from src.utils.PlotUtils import plot_hist2d, ScatterPlt
from src.datasets.PulseDataset import PulseDatasetWFPairNorm

def plot_2d_tensor(tensor, name, fname, logger : TensorBoardLogger):
    xaxis = arange(-0.5, tensor.shape[0] - 0.49, 1)
    yaxis = arange(-0.5, tensor.shape[1] - 0.49, 1)
    fig = plot_hist2d(xaxis, yaxis, tensor.detach().cpu().numpy(), name, "channel in dimension", "channel out dimension", "weight value", norm_to_bin_width=False, logz=False)
    logger.experiment.add_figure("model/{}".format(fname), fig)

def plot_1d_tensor(tensor, name, fname, logger : TensorBoardLogger):
    xaxis = [i for i in range(tensor.shape[0])]
    fig = ScatterPlt(xaxis, tensor.detach().cpu().numpy(), "sample", "height", title=name, marker='_')
    logger.experiment.add_figure("model/{}".format(fname), fig)

def gen_sample_data(config):
    ds = PulseDatasetWFPairNorm(config.dataset_config, "test",1,"cpu")
    (c,f), l = ds[0]
    for i in range(10):
        data = f[i].unsqueeze_(0)
        coo = c[i].unsqueeze_(0)
        torch.save([data,coo])



def plot_weights(model, logger):
    # extracting the model features at the particular layer number
    print(model)
    nums = []
    for layer_num in range(len(model)):
        layer = model[layer_num]
        # checking whether the layer is convolution layer or not
        if isinstance(layer, spconv.SparseConv2d):
            # getting the weight tensor data
            nums.append(layer_num)
            weight_tensor = model[layer_num].weight.data
            if weight_tensor.shape[0] == 1 and weight_tensor.shape[1] == 1:
                plot_2d_tensor(weight_tensor[0,0], "layer {}".format(layer_num), "layer_{}.png".format(layer_num), logger)
            print("conv2d layer - [{0} - {1}] {2}".format(model[layer_num].in_channels, model[layer_num].out_channels, model[layer_num].kernel_size))
        else:
            print(layer)
    return nums

def load_waveform(f):
    wf = torch.zeros((130,), dtype=float32)
    i = 0
    with open(f,'r') as csvfile:
        for l in csvfile.readlines():
            data = l.split(",")
            try:
                val = float(data[1].strip())
                wf[i] = val
                i += 1
            except Exception as e:
                pass
    return wf

def plot_waveform_evolution(model, layer_nums, wf, logger):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.features.detach()

        return hook
    for layer_num in layer_nums:
        model.model.model.network[layer_num].register_forward_hook(get_activation('layer_{}'.format(layer_num)))
    plot_1d_tensor(wf, "initial_waveform", "waveform", logger)
    coo = torch.tensor([8, 2, 0]).unsqueeze_(0)
    wf.unsqueeze_(0)
    #shape = torch.LongTensor([14,11])
    #t = SparseConvTensor(wf, coo, shape, 1)
    output = model([coo, wf])
    for layer_num in layer_nums:
        act = activation['layer_{}'.format(layer_num)].squeeze()
        if not act.shape:
            logger.experiment.add_scalar("model/final_wf_output", act.detach())
        if act.shape and act.shape[0] > 0:
            plot_1d_tensor(act, "after_layer_{}_waveform".format(layer_num), "waveform_layer_{}".format(layer_num), logger)
        #fig, axarr = plt.subplots(act.size(0))
        #for idx in range(act.size(0)):
        #    axarr[idx].imshow(act[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    parser.add_argument("checkpoint", help="path to checkpoint file")
    parser.add_argument("waveform_file", help="path to a file containing a waveform")
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
    model = modules.retrieve_class(config.run_config.run_class).load_from_checkpoint(args.checkpoint, config=config)
    model.eval()
    model.freeze()
    sparse_network = model.model.model.network
    layer_nums = plot_weights(sparse_network, logger)
    wf = load_waveform(args.waveform_file)
    plot_waveform_evolution(model, layer_nums, wf, logger)

    logger.finalize("done")
    logger.save()
    logger.close()


if __name__ == "__main__":
    main()