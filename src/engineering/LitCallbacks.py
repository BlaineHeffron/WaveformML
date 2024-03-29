import logging
from os.path import join

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.base import Callback
from torch import zeros
from src.utils.PlotUtils import plot_confusion_matrix, ScatterPlt
from torch.jit import script

class PSDCallbacks:

    def __init__(self, config):
        self.log = logging.getLogger(__name__)
        self.config = config
        self.callbacks = [EarlyStopping(monitor='val_loss', min_delta=.001, verbose=True, mode="min", patience=5)]
        self.callbacks.append(LoggingCallback())

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def set_args(self, args):
        # args["accumulate_grad_batches"] = {5: 2, 20: 3}
        if hasattr(self.config.optimize_config, "validation_freq"):
            if "check_val_every_n_epoch" not in args.keys():
                args["check_val_every_n_epoch"] = self.config.optimize_config.validation_freq
                self.log.info("Using a validation frequency of {}".format(self.config.optimize_config.validation_freq))
            else:
                self.log.info("Using a validation frequency of {}".format(args["check_val_every_n_epoch"]))
        return args


class LoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_loss = 100000

    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, "confusion_matrix"):
            pl_module.logger.experiment.add_figure("validation/confusion_matrix",
                                                   plot_confusion_matrix(
                                                       pl_module.confusion_matrix.detach().cpu().numpy(),
                                                       pl_module.config.system_config.type_names,
                                                       normalize=True))
            pl_module.confusion_matrix = zeros(pl_module.confusion_matrix.shape, device=pl_module.device)
        if not trainer.callback_metrics:
            return
        loss = trainer.callback_metrics["val_loss"].detach().item()
        if self.best_loss > loss:
            self.best_loss = loss
            pl_module.logger.log_hyperparams(pl_module.hparams, {"hp_metric": loss})

    def on_test_end(self, trainer, pl_module):
        if hasattr(pl_module, "test_confusion_matrix"):
            pl_module.logger.experiment.add_figure("evaluation/confusion_matrix_totals",
                                                   plot_confusion_matrix(
                                                       pl_module.test_confusion_matrix.detach().cpu().numpy().astype(
                                                           int),
                                                       pl_module.config.system_config.type_names,
                                                       normalize=False))
            pl_module.logger.experiment.add_figure("evaluation/confusion_matrix", plot_confusion_matrix(
                pl_module.test_confusion_matrix.detach().cpu().numpy(),
                pl_module.config.system_config.type_names,
                normalize=True))
            pl_module.test_confusion_matrix = zeros(pl_module.test_confusion_matrix.shape, device=pl_module.device)
        if hasattr(pl_module, "roc_curve"):
            res = pl_module.roc_curve.compute()
            class_name = pl_module.roc_curve.class_name
            nm = "evaluation/roc_curve"
            if class_name is not None:
                nm += "_{}".format(class_name)
            pl_module.logger.experiment.add_figure(nm, ScatterPlt(res[0], res[1],
                                                                  "true positive rate", "false positive rate"))
        pl_module.evaluator.dump()
