from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.engineering.LitCallbacks import LoggingCallback
from src.engineering.LitPSD import PSDDataModule
import argparse
from os.path import dirname, basename, join
from torch.jit import trace

from src.utils.util import get_config, ModuleUtility, get_tb_logdir_version, set_default_trainer_args, setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    parser.add_argument("checkpoint", help="path to checkpoint file")
    parser.add_argument("--calgroup", "-c", help="calibration group entry in PROSPECT_CALDB", type=str)
    parser.add_argument("--torchscript", "-t", action="store_true", help="set to generate torch script model instead of evaluating")
    parser.add_argument("--verbosity", "-v",
                        help="Set the verbosity for this run.",
                        type=int, default=0)
    args = parser.parse_args()
    main_logger = setup_logger(args)
    config = get_config(args.config)
    if args.calgroup:
        if hasattr(config.dataset_config, "calgroup"):
            print("Warning: overriding calgroup {0} with user supplied calgroup {1}".format(config.dataset_config.calgroup,args.calgroup))
        config.dataset_config.calgroup = args.calgroup
    log_folder = dirname(args.config)
    p = Path(log_folder)
    cp = p.glob('*.tfevents.*')
    logger = None
    if cp:
        for ckpt in cp:
            print("Using existing log file {}".format(ckpt))
            vnum = get_tb_logdir_version(str(ckpt))
            logger = TensorBoardLogger(dirname(dirname(log_folder)), name=basename(dirname(log_folder)), version=vnum)
            break
    else:
        logger = TensorBoardLogger(log_folder, name=config.run_config.exp_name)
        print("Creating new log file in directory {}".format(logger.log_dir))
    modules = ModuleUtility(config.run_config.imports)
    runner = modules.retrieve_class(config.run_config.run_class).load_from_checkpoint(args.checkpoint, config=config)
    trainer_args = {"logger": logger, "callbacks": [LoggingCallback()]}
    set_default_trainer_args(trainer_args, config)
    #model.set_logger(logger)
    if args.torchscript:
        runner.write_torchscript = True
    data_module = PSDDataModule(config, runner.device)
    trainer = Trainer(**trainer_args)
    trainer.test(runner, datamodule=data_module)

if __name__=="__main__":
    main()