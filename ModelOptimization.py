import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from LitCallbacks import *
from LitPSD import *
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import re
from util import save_config, DictionaryUtility, set_default_trainer_args

INDEX_PATTERN = re.compile(r'(\[[0-9]+\])')


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
        ind = int(m.group()[0][1:-1])
        if len(myobj) < ind + 1:
            raise IOError(
                "Optuna hyperparameter path config error: no object found at index {0} of {1}".format(ind, name))
        else:
            return myobj[ind]
    else:
        return getattr(obj, name)


def get_attribute(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    else:
        raise IOError(
            "optuna hyperparameter path not specified properly. {0} not found in {1}".format(name[0:m.start()], obj))


class ModelOptimization:
    """
    hyperparameter optimization class
    """

    def __init__(self, optuna_config, config, model_dir, trainer_args):
        self.optuna_config = optuna_config
        self.model_dir = model_dir
        self.config = config
        self.hyperparameters = {}
        base_dir = os.path.join(model_dir, "studies")
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        self.study_dir = os.path.join(model_dir, "studies/{}".format(config.run_config.exp_name))
        self.trainer_args = trainer_args
        if not os.path.exists(self.study_dir):
            os.mkdir(self.study_dir)
        self.hyperparameters_bounds = DictionaryUtility.to_dict(self.optuna_config.hyperparameters)
        self.parse_config()

    def parse_config(self):
        if not hasattr(self.optuna_config, "hyperparameters"):
            raise IOError(
                "No hyperparameters found in optuna config. You must set the hyperparameters to a dictionary of key: value where key is hte path to the hyperparameter in the config file, and value is an array of two elements bounding the range of the parameter")
        for h in self.hyperparameters_bounds.keys():
            i = 0
            for name in h.split("/"):
                if not name: 
                    continue
                if i > 0:
                    myobj = get_from_path(myobj, name)
                else:
                    myobj = get_from_path(self.config, name)
                i += 1
            #TODO: this doesnt actually let you modify the object, it will not work, DEBUG THIS
            self.hyperparameters[h] = myobj

    def modify_config(self, trial):
        for hp in self.hyperparameters.keys():
            name = hp.split("/")[-1]
            bounds = self.hyperparameters_bounds[hp]
            if isinstance(bounds[0], int):
                self.hyperparameters[hp] = trial.suggest_int(name, bounds[0], bounds[1])
            elif isinstance(bounds[0], float):
                self.hyperparameters[hp] = trial.suggest_float(name, bounds[0], bounds[1])
            elif isinstance(bounds[0], bool):
                self.hyperparameters[hp] = trial.suggest_int(name, 0, 1)

    def objective(self, trial):
        self.modify_config(trial)
        log_folder = os.path.join(self.study_dir, "runs")
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)
        logger = TensorBoardLogger(log_folder)
        psd_callbacks = PSDCallbacks(self.config)
        trainer_args = psd_callbacks.set_args(self.trainer_args)
        trainer_args["checkpoint_callback"] = \
            ModelCheckpoint(
                os.path.join(self.study_dir, "trial_{}".format(trial.number), "{epoch}"), monitor="val_acc")
        trainer_args["logger"] = logger
        trainer_args["default_root_dir"] = self.study_dir
        set_default_trainer_args(trainer_args, self.config)
        trainer_args["early_stop_callback"] = PyTorchLightningPruningCallback(trial, monitor="val_acc")
        save_config(self.config, log_folder, "trial_{}".format(trial.number), "config")
        #save_config(DictionaryUtility.to_object(trainer_args), log_folder,
        #        "trial_{}".format(trial.number), "train_args")
        metrics_callback = MetricsCallback()
        trainer = pl.Trainer(**trainer_args, callbacks=psd_callbacks.callbacks)
        model = LitPSD(self.config)
        trainer.fit(model)

        return metrics_callback.metrics[-1]["val_acc"].item()

    def run_study(self, pruning=False):
        pruner = optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(self.objective, **DictionaryUtility.to_dict(self.optuna_config.optimize_args))
        output = {}
        print("Number of finished trials: {}".format(len(study.trials)))
        output["n_finished_trials"] = len(study.trials)
        print("Best trial:")
        trial = study.best_trial
        output["best_trial"] = trial.value
        print("  Value: {}".format(trial.value))
        output["best_trial_params"] = trial.params
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        save_config(output, self.study_dir, "trial", "results", True)
