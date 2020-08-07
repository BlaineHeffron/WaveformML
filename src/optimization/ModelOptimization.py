import os
import re
import logging

from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.engineering.LitCallbacks import *
from src.engineering.LitPSD import *
from src.utils.util import save_config, DictionaryUtility, set_default_trainer_args, write_run_info

module_log = logging.getLogger(__name__)
INDEX_PATTERN = re.compile(r'\[([0-9]+)\]')


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def get_from_path(obj, name):
    m = INDEX_PATTERN.search(name)
    if m:
        myobj = getattr(obj, name[0:m.start()])
        for ind in m.group():
            ind = int(ind)
            if len(myobj) < ind + 1:
                raise IOError(
                    "Optuna hyperparameter path config error: no object found at index {0} of {1}".format(ind, name))
            else:
                myobj = myobj[ind]
        return myobj
    else:
        if not hasattr(obj, name):
            raise AttributeError(
                f"Object `{name}` cannot be loaded from `{obj}`."
            )
        return getattr(obj, name)


def get_attribute(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    else:
        raise IOError(
            "optuna hyperparameter path not specified properly. {0} not found in {1}".format(name, obj))


class ModelOptimization:
    """
    hyperparameter optimization class
    """

    def __init__(self, optuna_config, config, model_dir, trainer_args):
        self.optuna_config = optuna_config
        self.model_dir = model_dir
        self.config = config
        self.hyperparameters = {}
        self.log = logging.getLogger(__name__)
        base_dir = os.path.join(model_dir, "studies")
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        self.study_dir = os.path.join(model_dir, "studies/{}".format(config.run_config.exp_name))
        self.trainer_args = trainer_args
        if not os.path.exists(self.study_dir):
            os.mkdir(self.study_dir)
        write_run_info(self.study_dir)
        self.hyperparameters_bounds = DictionaryUtility.to_dict(self.optuna_config.hyperparameters)
        self.parse_config()

    def parse_config(self):
        if not hasattr(self.optuna_config, "hyperparameters"):
            raise IOError(
                "No hyperparameters found in optuna config. You must set the hyperparameters to a dictionary of key: value where key is hte path to the hyperparameter in the config file, and value is an array of two elements bounding the range of the parameter")
        for h in self.hyperparameters_bounds.keys():
            i = 0
            path_list = h.split("/")
            path_list = [p for p in path_list if p]
            plen = len(path_list)
            for j, name in enumerate(path_list):
                if not name:
                    continue
                if j == plen - 1:
                    break
                if i > 0:
                    myobj = get_from_path(myobj, name)
                else:
                    myobj = get_from_path(self.config, name)
                i += 1
            self.hyperparameters[h] = myobj

    def modify_config(self, trial):
        for hp in self.hyperparameters.keys():
            name = hp.split("/")[-1]
            bounds = self.hyperparameters_bounds[hp]
            if isinstance(bounds[0], int):
                setattr(self.hyperparameters[hp], name,
                        trial.suggest_int(name, bounds[0], bounds[1]))
            elif isinstance(bounds[0], float):
                setattr(self.hyperparameters[hp], name,
                        trial.suggest_float(name, bounds[0], bounds[1]))
            elif isinstance(bounds[0], bool):
                setattr(self.hyperparameters[hp], name,
                        trial.suggest_int(name, 0, 1))
            self.log.debug("setting {0} to {1}"
                           .format(hp, getattr(self.hyperparameters[hp], name)))

    def objective(self, trial):
        self.modify_config(trial)
        if not os.path.exists(self.study_dir):
            os.mkdir(self.study_dir)
        logger = TensorBoardLogger(self.study_dir, name="trial_{}".format(trial.number))
        log_folder = logger.log_dir
        if not os.path.exists(log_folder):
            os.makedirs(log_folder, exist_ok=True)
        psd_callbacks = PSDCallbacks(self.config)
        trainer_args = psd_callbacks.set_args(self.trainer_args)
        trainer_args["checkpoint_callback"] = \
            ModelCheckpoint(
                os.path.join(self.study_dir, "trial_{}".format(trial.number), "{epoch}"),
                monitor="val_acc")
        trainer_args["logger"] = logger
        trainer_args["default_root_dir"] = self.study_dir
        set_default_trainer_args(trainer_args, self.config)
        trainer_args["early_stop_callback"] = PyTorchLightningPruningCallback(trial, monitor="val_acc")
        if trainer_args["profiler"]:
            profiler = SimpleProfiler(output_filename=os.path.join(log_folder, "profile_results.txt"))
            trainer_args["profiler"] = profiler
        save_config(self.config, log_folder, "trial_{}".format(trial.number), "config")
        # save_config(DictionaryUtility.to_object(trainer_args), log_folder,
        #        "trial_{}".format(trial.number), "train_args")
        metrics_callback = MetricsCallback()
        cbs = psd_callbacks.callbacks
        cbs.append(metrics_callback)
        trainer = pl.Trainer(**trainer_args, callbacks=cbs)
        model = LitPSD(self.config)
        trainer.fit(model)
        return metrics_callback.metrics[-1]["val_checkpoint_on"].item()

    def run_study(self, pruning=False):
        pruner = optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="maximize", pruner=pruner)
        self.log.debug("optimize parameters: \n{}".format(DictionaryUtility.to_dict(self.optuna_config.optimize_args)))
        study.optimize(self.objective, **DictionaryUtility.to_dict(self.optuna_config.optimize_args), show_progress_bar=True, gc_after_trial=True)
        output = {}
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        self.log.info("Number of finished trials: {}".format(len(study.trials)))
        output["n_finished_trials"] = len(study.trials)
        self.log.info("Best trial:")
        output["best_trial"] = trial.value
        self.log.info("  Value: {}".format(trial.value))
        output["best_trial_params"] = trial.params
        self.log.info("  Params: ")
        for key, value in trial.params.items():
            self.log.info("    {}: {}".format(key, value))
        save_config(output, self.study_dir, "trial", "results", True)
