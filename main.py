import json
import logging
import util
import argparse
import os
from util import ModuleUtility, path_create

MODEL_DIR = "./model"
CONFIG_DIR = "./config"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="relative path of config file to use (in config folder)")
    parser.add_argument("--load_checkpoint", "-l",
                        help="load checkpoint based on experiment name and model name",
                        action="store_true")
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
    # read config
    with open(config_file) as json_data_file:
        config = json.load(json_data_file)
    # convert dict to object recursively for easy call
    config = util.DictionaryUtility.to_object(config)
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
