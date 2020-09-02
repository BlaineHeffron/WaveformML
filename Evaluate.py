from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.engineering.LitCallbacks import LoggingCallback
from src.engineering.LitPSD import LitPSD, PSDDataModule
import argparse
from os.path import dirname

from src.utils.util import get_config, ModuleUtility, get_tb_logdir_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    parser.add_argument("checkpoint", help="path to checkpoint file")
    args = parser.parse_args()
    config = get_config(args.config)
    log_folder = dirname(args.config)
    p = Path(log_folder)
    cp = p.glob('*.tfevents.*')
    logger = None
    if cp:
        for ckpt in cp:
            print("Using existing log file {}".format(ckpt))
            vnum = get_tb_logdir_version(str(ckpt))
            logger = TensorBoardLogger(dirname(log_folder), name=config.run_config.exp_name, version=vnum)
            break
    else:
        logger = TensorBoardLogger(log_folder, name=config.run_config.exp_name)
        print("Creating new log file in directory {}".format(logger.log_dir))
    modules = ModuleUtility(config.run_config.imports)
    runner = modules.retrieve_class(config.run_config.run_class).load_from_checkpoint(args.checkpoint, config)
    trainer_args = {"logger": logger}
    trainer_args["callbacks"] = [LoggingCallback()]
    model = LitPSD.load_from_checkpoint(args.checkpoint, config)
    #model.set_logger(logger)
    data_module = PSDDataModule(config, runner.device)

    trainer = Trainer(**trainer_args)
    trainer.test(model, datamodule=data_module)

if __name__=="__main__":
    main()