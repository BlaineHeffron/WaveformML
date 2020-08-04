from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from LitCallbacks import *
from os.path import join, abspath, exists
from LitPSD import *

import json
import logging
import util
import argparse
import os
from util import path_create, ValidateUtility

MODEL_DIR = "./model"
CONFIG_DIR = "./config"
CONFIG_VALIDATION = "./config_requirements.json"


def save_path(model_folder, model_name, exp_name):
    if exp_name:
        if model_name: return join(model_folder, model_name + "_" + exp_name + "_{epoch:02d}-{val_loss:.2f}")
        else:
            raise IOError("No model name given. Set model_name property in net_config.")
    else:
        raise IOError("No experiment name given. Set exp_name property in run_config.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="relative path of config file to use (in config folder)")
    parser.add_argument("--load_checkpoint", "-l",
                        help="Load checkpoint based on experiment name and model name",
                        action="store_true")
    parser.add_argument("--name", "-n",
                        help="Set the experiment name for this run. Overrides exp_name specified in the run_config.",
                        type=str)
    # TODO implement verbosity
    parser.add_argument("--verbosity", "-v",
                        help="Set the verbosity for this run.",
                        type=int)
    parser.add_argument("--validation", "-cv", type=str,
                        help="Set the path to the config validation file.", )
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument()
    args = parser.parse_args()
    if not os.path.exists(MODEL_DIR):
        path_create(MODEL_DIR)
    config_file = args.config
    if not config_file.endswith(".json"):
        config_file = "{}.json".format(config_file)
    if not os.path.isabs(config_file):
        config_file = os.path.join(CONFIG_DIR, config_file)
        if not os.path.exists(config_file):
            config_file = os.path.join(os.getcwd(), config_file)
            if not os.path.exists(config_file):
                raise IOError("Could not find config file {0}. search in"
                              " {1}".format(args.config, config_file))
    if args.validation:
        valid_file = args.validation
    else:
        valid_file = CONFIG_VALIDATION
    # read config
    with open(config_file) as json_data_file:
        config = json.load(json_data_file)
    # validate config
    if not os.path.exists(valid_file):
        print("WARNING: Could not find config validation file. Search path is set to {}".format(CONFIG_VALIDATION))
    else:
        ValidateUtility.validate_config(config, valid_file)
    # convert dict to object recursively for easy call
    config = util.DictionaryUtility.to_object(config)
    if not hasattr(config, "system_config"): raise IOError("Config file must contain system_config")
    if not hasattr(config, "dataset_config"): raise IOError("Config file must contain dataset_config")
    if not hasattr(config.dataset_config, "paths"): raise IOError("Dataset config must contain paths list")
    if hasattr(config.system_config, "model_name"):
        model_name = config.system_config.model_name
    else:
        model_name = util.unique_path_combine(config.dataset_config.paths)
    model_folder = join(abspath("./model"), model_name)
    if hasattr(config, "run_config"):
        if not hasattr(config.run_config, "exp_name"):
            counter = 1
            exp_name = "experiment_{0}".format(counter)
            while exists(save_path(model_folder, model_name, exp_name)):
                counter += 1
                exp_name = "experiment_{0}".format(counter)
            config.run_config.exp_name = exp_name
    if args.name:
        config.run_config.exp_name = args.name
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-6s %(message)s')

    logging.info('=======================================================')
    logging.info('Using system from %s' % config_file)
    logging.info('=======================================================')

    log_folder = join(model_folder, "runs")
    model = LitPSD(config)
    data_module = PSDDataModule(config.dataset_config)
    logger = TensorBoardLogger(log_folder, name=model_name)
    psd_callbacks = PSDCallbacks(config)
    trainer_args = psd_callbacks.set_args(args)
    trainer_args["checkpoint_callback"] = \
        ModelCheckpoint(
            filepath=save_path(model_folder, model_name, config.run_config.exp_name))
    trainer_args["logger"] = logger
    trainer_args["default_root_dir"] = model_folder
    trainer_args["precision"] = 16
    if hasattr(config.system_config, "gpu_enabled"):
        if config.system_config.gpu_enabled:
            if not args["gpus"]:
                args["gpus"] = 1
            if args["num_nodes"] > 1:
                trainer_args["distributed_backend"] = 'ddp'
    trainer = Trainer(**trainer_args, callbacks=psd_callbacks.callbacks)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())


if __name__ == '__main__':
    main()
