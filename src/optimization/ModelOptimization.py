import os
import re
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler
import optuna

from src.engineering.LitCallbacks import *
from src.engineering.LitPSD import *
from src.utils.util import ModuleUtility
from src.utils.util import save_config, DictionaryUtility, set_default_trainer_args, write_run_info

module_log = logging.getLogger(__name__)
INDEX_PATTERN = re.compile(r'\[([0-9]+)\]')


class PruningCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.callback_metrics:
            val = trainer.callback_metrics["val_loss"].detach().item()
            if not hasattr(pl_module, "trial"):
                raise Exception("No Trial found in lightning module {}".format(pl_module))
            pl_module.trial.report(val, batch_idx)
            prune = False
            try:
                prune = pl_module.trial.should_prune()
            except Exception as e: 
                print(e)
            if prune: 
                raise optuna.TrialPruned()


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
            os.makedirs(base_dir, exist_ok=True)
        self.study_dir = os.path.join(model_dir, "studies/{}".format(config.run_config.exp_name))
        self.study_name = self.config.run_config.exp_name if not hasattr(optuna_config,
                                                                         "name") else self.optuna_config.name
        self.trainer_args = trainer_args
        if not os.path.exists(self.study_dir):
            os.makedirs(self.study_dir, exist_ok=True)
        self.connstr = "sqlite:///" + os.path.join(self.study_dir, "study.db")
        write_run_info(self.study_dir)
        self.hyperparameters_bounds = DictionaryUtility.to_dict(self.optuna_config.hyperparameters)
        self.log.debug("hyperparameters bounds set to {0}".format(self.hyperparameters_bounds))
        self.modules = ModuleUtility(["optuna.pruners", "optuna.samplers"])
        self.parse_config()

    def parse_config(self):
        if not hasattr(self.optuna_config, "hyperparameters"):
            raise IOError(
                "No hyperparameters found in optuna config. You must set the hyperparameters to a dictionary of key: "
                "value where key is hte path to the hyperparameter in the config file, and value is an array of two "
                "elements bounding the range of the parameter")
        for h in self.hyperparameters_bounds.keys():
            i = 0
            path_list = h.split("/")
            path_list = [p for p in path_list if p]
            plen = len(path_list)
            myobj = None
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
            if myobj:
                self.hyperparameters[h] = myobj

    def modify_config(self, trial):
        for hp in self.hyperparameters.keys():
            name = hp.split("/")[-1]
            bounds = self.hyperparameters_bounds[hp]
            if isinstance(bounds, dict):
                if "val" in bounds.keys():
                    setattr(self.hyperparameters[hp], name,
                            trial.suggest_categorical(name, bounds["val"]))
                else:
                    raise ValueError("Invalid format for hyperparameter key {0}. Specify category with \"val\":[list "
                                     "of values]".format(hp))
            elif len(bounds) > 2:
                setattr(self.hyperparameters[hp], name,
                        trial.suggest_categorical(name, bounds))
            elif isinstance(bounds[0], int):
                setattr(self.hyperparameters[hp], name,
                        trial.suggest_int(name, bounds[0], bounds[1]))
            elif isinstance(bounds[0], float):
                t = None
                if bounds[0] != 0 and bounds[1] != 0:
                    if bounds[1] / bounds[0] > 100 or bounds[0] / bounds[1] > 100:
                        t = trial.suggest_loguniform(name, bounds[0], bounds[1])
                if t is None:
                    t = trial.suggest_float(name, bounds[0], bounds[1])
                setattr(self.hyperparameters[hp], name, t)
            elif isinstance(bounds[0], bool):
                setattr(self.hyperparameters[hp], name,
                        trial.suggest_int(name, 0, 1))
            self.log.debug("setting {0} to {1}"
                           .format(hp, getattr(self.hyperparameters[hp], name)))

    def objective(self, trial):
        self.modify_config(trial)
        if not os.path.exists(self.study_dir):
            os.mkdir(self.study_dir)
        if not os.path.exists(os.path.join(self.study_dir, "trial_{}".format(trial.number))):
            os.mkdir(os.path.join(self.study_dir, "trial_{}".format(trial.number)))
        logger = TensorBoardLogger(self.study_dir, name="trial_{}".format(trial.number), default_hp_metric=False)
        log_folder = logger.log_dir
        if not os.path.exists(log_folder):
            os.makedirs(log_folder, exist_ok=True)
        trainer_args = self.trainer_args
        checkpoint_callback = \
            ModelCheckpoint(
                dirpath=log_folder, filename='{epoch}-{val_loss:.2f}',
                monitor="val_loss")
        trainer_args["logger"] = logger
        trainer_args["default_root_dir"] = self.study_dir
        set_default_trainer_args(trainer_args, self.config)
        if trainer_args["profiler"]:
            profiler = SimpleProfiler(output_filename=os.path.join(log_folder, "profile_results.txt"))
            trainer_args["profiler"] = profiler
        save_config(self.config, log_folder, "trial_{}".format(trial.number), "config")
        # save_config(DictionaryUtility.to_object(trainer_args), log_folder,
        #        "trial_{}".format(trial.number), "train_args")
        metrics_callback = MetricsCallback()
        cbs = [metrics_callback]
        cbs.append(LoggingCallback())
        cbs.append(PruningCallback())
        cbs.append(checkpoint_callback)
        # trainer_args["early_stop_callback"] = PyTorchLightningPruningCallback(trial, monitor="val_early_stop_on")
        if self.config.run_config.run_class == "LitZ":
            cbs.append(EarlyStopping(monitor='val_loss', min_delta=.00, verbose=True, mode="min", patience=5))
        else:
            cbs.append(EarlyStopping(monitor='val_loss', min_delta=.00, verbose=True, mode="min", patience=4))

        trainer = pl.Trainer(**trainer_args, callbacks=cbs)
        modules = ModuleUtility(self.config.run_config.imports)
        model = modules.retrieve_class(self.config.run_config.run_class)(self.config, trial)
        data_module = PSDDataModule(self.config, model.device)
        trainer.fit(model, datamodule=data_module)
        if metrics_callback.metrics:
            return metrics_callback.metrics[-1]["val_loss"].detach().item()
        else:
            return 0

    def run_study(self, pruning=False):
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10,
                                             interval_steps=3) if pruning else optuna.pruners.NopPruner()
        if hasattr(self.optuna_config, "pruner"):
            if hasattr(self.optuna_config, "pruner_params"):
                pruner = self.modules.retrieve_class("pruners." + self.optuna_config.pruner)(
                    **DictionaryUtility.to_dict(self.optuna_config.pruner_params))
            else:
                pruner = self.modules.retrieve_class("pruners." + self.optuna_config.pruner)()
        opt_dict = {}
        if hasattr(self.optuna_config, "sampler"):
            if hasattr(self.optuna_config, "sampler_params"):
                opt_dict["sampler"] = self.modules.retrieve_class("samplers." + self.optuna_config.sampler)(
                    **DictionaryUtility.to_dict(self.optuna_config.sampler_params))
            else:
                opt_dict["sampler"] = self.modules.retrieve_class("samplers." + self.optuna_config.sampler)()

        study = optuna.create_study(study_name=self.study_name, direction="minimize", pruner=pruner,
                                    storage=self.connstr, load_if_exists=True, **opt_dict)
        self.log.debug("optimize parameters: \n{}".format(DictionaryUtility.to_dict(self.optuna_config.optimize_args)))
        study.optimize(self.objective, **DictionaryUtility.to_dict(self.optuna_config.optimize_args),
                       show_progress_bar=True, gc_after_trial=True)
        output = {}
        self.log.info("Number of finished trials: {}".format(len(study.trials)))
        self.log.info("Best trial:")
        trial = study.best_trial
        self.log.info("  Value: {}".format(trial.value))
        self.log.info("  Params: ")
        for key, value in trial.params.items():
            self.log.info("    {}: {}".format(key, value))
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
