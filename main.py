from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler
from os.path import exists, join
from src.optimization.ModelOptimization import *
from src.utils.ModelValidation import *

import sys
import json
import logging
from src.utils import util
import argparse
from src.utils.util import  ValidateUtility, save_config, save_path, set_default_trainer_args, \
    retrieve_model_checkpoint, get_tb_logdir_version, check_config, setup_logger, get_model_folder

MODEL_DIR = "./model"
CONFIG_DIR = "./config"
CONFIG_VALIDATION = "./config_requirements.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="relative path of config file to use (in config folder)")
    parser.add_argument("--name", "-n",
                        help="Set the experiment name for this run. Overrides exp_name specified in the run_config.",
                        type=str)
    # TODO implement verbosity
    parser.add_argument("--load_best", "-lb", action="store_true",
                        help="finds the best checkpoint matching the model and experiment names, loads it and resumes training.")
    parser.add_argument("--load_checkpoint", "-l", type=str,
                        help="Set the path to the checkpoint you'd like to resume training on.")
    parser.add_argument("--restore_training", "-r", action="store_true",
                        help="Restores the training state in addition to model weights when loading a checkpoint. Does nothing if no checkpoints are loaded.")
    parser.add_argument("--test", "-t", action="store_true", help="Run test on model after training.")
    parser.add_argument("--verbosity", "-v",
                        help="Set the verbosity for this run.",
                        type=int, default=0)
    parser.add_argument("--logfile", "-lf",
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
    non_trainer_args = ["config", "load_checkpoint", "load_best", "name",
                        "restore_training", "verbosity",
                        "logfile", "config_validation", "optimize_config",
                        "pruning", "validate", "test"]
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    run_test = False
    if args.test:
        run_test = True
    verbosity = args.verbosity
    config_file = args.config
    config_file = check_config(config_file, CONFIG_DIR)
    #if args.config_validation:
    #    valid_file = args.config_validation
    #else:
    #    valid_file = CONFIG_VALIDATION
    # read config
    with open(config_file) as json_data_file:
        config = json.load(json_data_file)
    # validate config
    #if not os.path.exists(valid_file):
    #    print("WARNING: Could not find config validation file. Search path is set to {}".format(CONFIG_VALIDATION))
    #else:
    #    ValidateUtility.validate_config(config, valid_file)
    # convert dict to object recursively for easy call
    config = util.DictionaryUtility.to_object(config)
    if not hasattr(config, "system_config"): raise IOError("Config file must contain system_config")
    if not hasattr(config, "dataset_config"): raise IOError("Config file must contain dataset_config")
    if not hasattr(config.dataset_config, "paths"): raise IOError("Dataset config must contain paths list")
    model_name, model_folder = get_model_folder(config)
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
    main_logger = setup_logger(args)
    logging.debug('Command line arguments: %s' % str(sys.argv))

    logging.info('=======================================================')
    logging.info('Using system from %s' % config_file)
    logging.info('=======================================================')

    if args.validate:
        ModelValidation.validate(config)

    if args.auto_lr_find:
        setattr(args, "auto_lr_find", True)

    if args.optimize_config or hasattr(config, "optuna_config"):
        set_pruning = args.pruning
        opt_config = args.optimize_config
        trainer_args = vars(args)
        for non_trainer_arg in non_trainer_args:
            del trainer_args[non_trainer_arg]
        if opt_config:
            logging.info('Running optimization routine using optuna config file: %s' % str(opt_config))
            opt_config = check_config(opt_config, CONFIG_DIR)
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
        exp_folder = join(tb_folder, config.run_config.exp_name)
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)
        load_checkpoint = None
        if args.load_best:
            load_checkpoint = retrieve_model_checkpoint(exp_folder, model_name, config.run_config.exp_name)
        if args.load_checkpoint:
            load_checkpoint = args.load_checkpoint
        if args.restore_training and load_checkpoint:
            vnum = get_tb_logdir_version(load_checkpoint)
            if vnum:
                logger = TensorBoardLogger(tb_folder, name=config.run_config.exp_name, version=vnum, default_hp_metric=False)
                main_logger.info("Utilizing existing log directory {}".format(logger.log_dir))
            else:
                logger = TensorBoardLogger(tb_folder, name=config.run_config.exp_name, default_hp_metric=False)
        else:
            logger = TensorBoardLogger(tb_folder, name=config.run_config.exp_name, default_hp_metric=False)
        log_folder = logger.log_dir
        if not os.path.exists(log_folder):
            os.makedirs(log_folder, exist_ok=True)
        write_run_info(log_folder)
        psd_callbacks = PSDCallbacks(config)
        trainer_args = vars(args)
        for non_trainer_arg in non_trainer_args:
            if non_trainer_arg == "restore_training" and load_checkpoint:
                main_logger.info("Training is set to resume from model checkpoint {}".format(load_checkpoint))
                trainer_args["resume_from_checkpoint"] = load_checkpoint
            del trainer_args[non_trainer_arg]
        trainer_args = psd_callbacks.set_args(trainer_args)
        checkpoint_callback = \
            ModelCheckpoint(
                dirpath=log_folder,
                filename='{epoch}-{val_loss:.2f}',
                monitor="val_loss")
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
        if load_checkpoint:
            main_logger.info("Loading model checkpoint {}".format(load_checkpoint))
            runner = modules.retrieve_class(config.run_config.run_class).load_from_checkpoint(load_checkpoint, config)
        else:
            runner = modules.retrieve_class(config.run_config.run_class)(config)
        data_module = PSDDataModule(config, runner.device)
        psd_callbacks.add_callback(checkpoint_callback)
        trainer = Trainer(**trainer_args, callbacks=psd_callbacks.callbacks)
        trainer.fit(runner, datamodule=data_module)
        if run_test:
            trainer.test()


if __name__ == '__main__':
    main()
