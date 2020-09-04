import logging
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks.base import Callback
from torch import zeros
from src.utils.PlotUtils import plot_confusion_matrix


class PSDCallbacks:

    def __init__(self, config):
        self.log = logging.getLogger(__name__)
        self.config = config
        self.callbacks = [EarlyStopping(min_delta=.00, verbose=True, mode="min", patience=10)]
        self.callbacks.append(LearningRateLogger())
        self.callbacks.append(LoggingCallback())

    def set_args(self, args):
        # args["accumulate_grad_batches"] = {5: 2, 20: 3}
        if hasattr(self.config.optimize_config, "validation_freq"):
            if not "check_val_every_n_epoch" in args.keys():
                args["check_val_every_n_epoch"] = self.config.optimize_config.validation_freq
                self.log.info("Using a validation frequency of {}".format(self.config.optimize_config.validation_freq))
            else:
                self.log.info("Using a validation frequency of {}".format(args["check_val_every_n_epoch"]))
        return args


class LoggingCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, "confusion_matrix"):
            pl_module.logger.experiment.add_figure("validation/confusion_matrix", plot_confusion_matrix(pl_module.confusion_matrix.detach().cpu().numpy(),
                                                                                        pl_module.config.system_config.type_names,
                                                                                        normalize=True))
            pl_module.confusion_matrix = zeros(pl_module.confusion_matrix.shape, device=pl_module.device)
        #if hasattr(pl_module.config.net_config, "hparams"):
        #    pl_module.logger.experiment.add_hparams(flatten(pl_module.hparams["net_config"]["hparams"]), {"epoch_val_acc": 1., "epoch_val_loss": 1.})

    def on_test_epoch_end(self, trainer, pl_module):
        pl_module.logger.experiment.add_figure("evaluation/confusion_matrix_totals", plot_confusion_matrix(pl_module.test_confusion_matrix.detach().cpu().numpy(),
                                                                                                    pl_module.config.system_config.type_names,
                                                                                                    normalize=False))
        pl_module.logger.experiment.add_figure("evaluation/confusion_matrix", plot_confusion_matrix(pl_module.test_confusion_matrix.detach().cpu().numpy(),
                                                             pl_module.config.system_config.type_names,
                                                             normalize=True))
        pl_module.test_confusion_matrix = zeros(pl_module.test_confusion_matrix.shape, device=pl_module.device)
        pl_module.evaluator.dump()
