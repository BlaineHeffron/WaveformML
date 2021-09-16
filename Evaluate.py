from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from main import choose_data_module
from src.engineering.GraphDataModule import GraphDataModule
from src.engineering.LitCallbacks import LoggingCallback
from src.engineering.LitPSD import PSDDataModule
import argparse
from os.path import dirname, basename
import torch

from src.utils.util import get_config, ModuleUtility, get_tb_logdir_version, set_default_trainer_args, setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    parser.add_argument("checkpoint", help="path to checkpoint file")
    parser.add_argument("--calgroup", "-c", help="calibration group entry in PROSPECT_CALDB", type=str)
    parser.add_argument("--onnx", "-o", action="store_true", help="set to generate onnx model instead of evaluating")
    parser.add_argument("--occlude", "-oc", type=int, default=-1, help="feature index to zero out during evaluation")
    parser.add_argument("--num_threads", "-nt", type=int, help="number of threads to use")
    parser.add_argument("--verbosity", "-v",
                        help="Set the verbosity for this run.",
                        type=int, default=0)
    args = parser.parse_args()
    main_logger = setup_logger(args)
    config = get_config(args.config)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    if args.calgroup:
        if hasattr(config.dataset_config, "calgroup"):
            print("Warning: overriding calgroup {0} with user supplied calgroup {1}".format(config.dataset_config.calgroup,args.calgroup))
        config.dataset_config.calgroup = args.calgroup
    log_folder = dirname(args.config)
    p = Path(log_folder)
    cp = p.glob('*.tfevents.*')
    tb_logger_args = {}
    if args.occlude > -1:
        tb_logger_args = {"sub_dir": "occlude_{0}".format(args.occlude)}
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
    if args.occlude > -1:
        setattr(config.dataset_config, "occlude_index", args.occlude)
    modules = ModuleUtility(config.run_config.imports)
    runner = modules.retrieve_class(config.run_config.run_class).load_from_checkpoint(args.checkpoint, config=config)
    trainer_args = {"logger": logger, "callbacks": [LoggingCallback()]}
    set_default_trainer_args(trainer_args, config)
    #model.set_logger(logger)
    if args.onnx:
        runner.write_onnx = True
    data_module = choose_data_module(config, runner.device)
    trainer = Trainer(**trainer_args)
    trainer.test(runner, datamodule=data_module)

if __name__=="__main__":
    main()