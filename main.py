from pytorch_lightning.profiler import AdvancedProfiler
from os.path import abspath, exists
from src.optimization.ModelOptimization import *

import json
import logging
from src.utils import util
import argparse
import os
from src.utils.util import path_create, ValidateUtility, save_config, save_path, set_default_trainer_args

MODEL_DIR = "./model"
CONFIG_DIR = "./config"
CONFIG_VALIDATION = "./config_requirements.json"


def check_config(config_file):
    orig = config_file
    if not config_file.endswith(".json"):
        config_file = "{}.json".format(config_file)
    if not os.path.isabs(config_file):
        config_file = os.path.join(CONFIG_DIR, config_file)
        if not os.path.exists(config_file):
            config_file = os.path.join(os.getcwd(), config_file)
            if not os.path.exists(config_file):
                raise IOError("Could not find config file {0}. search in"
                              " {1}".format(orig, config_file))
    return config_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="relative path of config file to use (in config folder)")
    parser.add_argument("--name", "-n",
                        help="Set the experiment name for this run. Overrides exp_name specified in the run_config.",
                        type=str)
    # TODO implement verbosity
    parser.add_argument("--verbosity", "-v",
                        help="Set the verbosity for this run.",
                        type=int, default=0)
    parser.add_argument("--logfile", "-l",
                        help="Set the filename or path to the filename for the program log this run."
                             " Set --verbosity to control the amount of information logged.",
                        type=str)
    parser.add_argument("--validation", "-cv", type=str,
                        help="Set the path to the config validation file.")
    parser.add_argument("--optimize_config", "-oc", type=str,
                        help="Set the path to the optuna optimization config file.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
             "trials at the early stages of training.",
    )
    non_trainer_args = ["config", "name", "verbosity",
                        "logfile", "validation", "optimize_config",
                        "pruning"]
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    verbosity = args.verbosity
    if not os.path.exists(MODEL_DIR):
        path_create(MODEL_DIR)
    config_file = args.config
    config_file = check_config(config_file)
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
    debug_level = logging.NOTSET
    if verbosity:
        if verbosity == 1:
            debug_level = logging.CRITICAL
        elif verbosity == 2:
            debug_level = logging.ERROR
        elif verbosity == 3:
            debug_level = logging.WARNING
        elif verbosity == 4:
            debug_level = logging.INFO
        elif verbosity == 5:
            debug_level = logging.DEBUG

    loggingargs = {}
    if args.logfile:
        loggingargs["filename"] = args.logfile

    logging.basicConfig(level=debug_level,
                        format='%(levelname)-6s %(message)s',
                        **loggingargs)

    logging.info('=======================================================')
    logging.info('Using system from %s' % config_file)
    logging.info('=======================================================')

    if args.optimize_config or hasattr(config, "optuna_config"):
        set_pruning = args.pruning
        opt_config = args.optimize_config
        trainer_args = vars(args)
        for non_trainer_arg in non_trainer_args:
            del trainer_args[non_trainer_arg]
        if opt_config:
            opt_config = check_config(opt_config)
            with open(opt_config) as f:
                opt_config = json.load(f)
                opt_config = util.DictionaryUtility.to_object(opt_config)
            m = ModelOptimization(opt_config, config, model_folder, trainer_args)
        else:
            m = ModelOptimization(config.optuna_config, config, model_folder, trainer_args)
        m.run_study(pruning=set_pruning)
    else:
        tb_folder = join(model_folder, "runs")
        if not os.path.exists(tb_folder):
            os.mkdir(tb_folder)
        logger = TensorBoardLogger(tb_folder, name=config.run_config.exp_name)
        log_folder = logger.log_dir
        if not os.path.exists(log_folder):
            os.makedirs(log_folder, exist_ok=True)
        psd_callbacks = PSDCallbacks(config)
        trainer_args = vars(args)
        for non_trainer_arg in non_trainer_args:
            del trainer_args[non_trainer_arg]
        trainer_args = psd_callbacks.set_args(trainer_args)
        trainer_args["checkpoint_callback"] = \
            ModelCheckpoint(
                filepath=save_path(model_folder, model_name, config.run_config.exp_name))
        if trainer_args["profiler"] or verbosity >= 5:
            if verbosity >= 5:
                profiler = AdvancedProfiler(output_filename=join(log_folder, "profile_results.txt"))
            else:
                profiler = SimpleProfiler(output_filename=join(log_folder, "profile_results.txt"))
            trainer_args["profiler"] = profiler
        trainer_args["logger"] = logger
        trainer_args["default_root_dir"] = model_folder
        set_default_trainer_args(trainer_args, config)
        save_config(config, log_folder, config.run_config.exp_name, "config")
        # save_config(DictionaryUtility.to_object(trainer_args), log_folder,
        #        config.run_config.exp_name, "train_args")
        model = LitPSD(config)
        trainer = Trainer(**trainer_args, callbacks=psd_callbacks.callbacks)
        trainer.fit(model)


if __name__ == '__main__':
    main()
