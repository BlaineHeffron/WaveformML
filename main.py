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
                raise IOError("Could not find config file {0}.".format(args.config))
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

    modules = ModuleUtility(config.run_config.imports)
    runner = modules.retrieve_class(config.run_config.run_class)(config, args.load_checkpoint)
    runner.run()


if __name__ == '__main__':
    main()
