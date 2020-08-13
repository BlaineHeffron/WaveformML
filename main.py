from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler
from os.path import abspath, exists, join
from src.optimization.ModelOptimization import *
from src.utils.ModelValidation import *

import sys
import json
import logging
from src.utils import util
import argparse
from src.utils.util import path_create, ValidateUtility, save_config, save_path, set_default_trainer_args

MODEL_DIR = "./model"
CONFIG_DIR = "./config"
CONFIG_VALIDATION = "./config_requirements.json"


def setup_logger(args):
    debug_level = logging.ERROR
    if args.verbosity:
        if args.verbosity == 1:
            debug_level = logging.CRITICAL
        elif args.verbosity == 2:
            debug_level = logging.ERROR
        elif args.verbosity == 3:
            debug_level = logging.WARNING
        elif args.verbosity == 4:
            debug_level = logging.INFO
        elif args.verbosity == 5:
            debug_level = logging.DEBUG

    # set up logging to file - see previous section for more details
    logargs = {}
    if args.logfile:
        logargs["filename"] = args.logfile
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filemode='a', **logargs)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # create logger
    logger = logging.getLogger('WaveformML')
    logger.setLevel(debug_level)

    logger.info("logging verbosity set to {}".format(args.verbosity))

    # create formatter
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    # create console handler and set level to  user specified level
    # ch = logging.StreamHandler()
    # ch.setLevel(debug_level)
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)

    # create a file handler if specified in command args
    if args.logfile:
        fh = logging.FileHandler(args.logfile)
        fh.setLevel(debug_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Logging to file {}".format(args.logfile))


def check_config(config_file):
    orig = config_file
    if not config_file.endswith(".json"):
        config_file = "{}.json".format(config_file)
    if not os.path.isabs(config_file):
        config_file = join(CONFIG_DIR, config_file)
        if not os.path.exists(config_file):
            config_file = join(os.getcwd(), config_file)
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
    parser.add_argument("--validate", "-va", action="store_true",
                        help="If set, will validate the input algorithm before running")
    parser.add_argument("--optimize_config", "-oc", type=str,
                        help="Set the path to the optuna optimization config file.")
    parser.add_argument("--config_validation", "-cv", type=str,
                        help="Set the path to the config validation file.")
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
    config_file = args.config
    config_file = check_config(config_file)
    if args.config_validation:
        valid_file = args.config_validation
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
    if hasattr(config.system_config, "model_base_path"):
        model_dir = config.system_config.model_base_path
    else:
        model_dir = MODEL_DIR
        setattr(config.system_config, "model_base_path", model_dir)
    if not os.path.exists(model_dir):
        path_create(model_dir)
    model_folder = join(abspath(model_dir), model_name)
    if not os.path.exists(model_folder):
        path_create(model_folder)
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
    setup_logger(args)
    logging.debug('Command line arguments: %s' % str(sys.argv))

    logging.info('=======================================================')
    logging.info('Using system from %s' % config_file)
    logging.info('=======================================================')

    if args.validate:
        ModelValidation.validate(config)

    if args.optimize_config or hasattr(config, "optuna_config"):
        set_pruning = args.pruning
        opt_config = args.optimize_config
        trainer_args = vars(args)
        for non_trainer_arg in non_trainer_args:
            del trainer_args[non_trainer_arg]
        if opt_config:
            logging.info('Running optimization routine using optuna config file: %s' % str(opt_config))
            opt_config = check_config(opt_config)
            with open(opt_config) as f:
                opt_config = json.load(f)
                opt_config = util.DictionaryUtility.to_object(opt_config)
            m = ModelOptimization(opt_config, config, model_folder, trainer_args)
        else:
            logging.info('Running optimization routine using optuna_config')
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
        write_run_info(log_folder)
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

        modules = ModuleUtility(config.run_config.imports)
        runner = modules.retrieve_class(config.run_config.run_class)(config)
        data_module = PSDDataModule(config, runner.device)
        trainer = Trainer(**trainer_args, callbacks=psd_callbacks.callbacks)
        trainer.fit(runner, data_module)


if __name__ == '__main__':
    main()
