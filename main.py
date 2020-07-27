import json
import logging
import util
from net import *
import argparse

LOG_DIR = "./logs"
MODEL_DIR = "./model"
OUT_DIR = "./results"
CONFIG_DIR = "./config"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="relative path of config file to use (in configs folder)")
    args = parser.parse_args()
    config_file = args.config
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    # read config
    with open(config_file) as json_data_file:
        config = json.load(json_data_file)
    # convert dict to object recursively for easy call
    config = util.DictionaryUtility.to_object(config)
    exp_name = config.run_config.exp_name
    path_prefix = os.path.join(LOG_DIR, exp_name)
    if not os.path.exists('./model/' + exp_name):
        os.mkdir('./model/' + exp_name)
    # save config for record
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(util.DictionaryUtility.to_dict(config), outfile, indent=2)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-6s %(message)s')

    logging.info('=======================================================')
    logging.info('Using system from %s' % config_file)
    logging.info('=======================================================')

    modules = ModuleUtility(config.run_config.imports)
    runner = modules.retrieve_class(config.run_config.run_class)(config)
    runner.run()


if __name__ == '__main__':
    main()
