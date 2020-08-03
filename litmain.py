from pytorch_lightning import Trainer
from LitPSD import *

import json
import logging
import util
import argparse
import os
from util import ModuleUtility, path_create, ValidateUtility

MODEL_DIR = "./model"
CONFIG_DIR = "./config"
CONFIG_VALIDATION = "./config_requirements.json"


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
    parser.add_argument("--nodes", "-no",
                        help="Set the number of nodes used for this run.", default=1,
                        type=int)
    parser.add_argument("--validation", "-cv", type=str,
                        help="Set the path to the config validation file.",)
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
                        " {1}".format(args.config,config_file))
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
    if args.name:
        config.run_config.exp_name = args.name
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-6s %(message)s')

    logging.info('=======================================================')
    logging.info('Using system from %s' % config_file)
    logging.info('=======================================================')

    model = LitPSD(config)
    data_module = PSDDataModule(config.dataset_config)
    if hasattr(config.system_config,"gpu_enabled"):
        if config.system_config.gpu_enabled:
            trainer = Trainer(gpus=1, num_nodes=args.nodes, distributed_backend='ddp')
    else:
        trainer = Trainer(num_nodes=args.nodes)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

if __name__ == '__main__':
    main()
