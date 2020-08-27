import logging
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks.base import Callback


class PSDCallbacks:

    def __init__(self, config):
        self.config = config
        self.callbacks = [EarlyStopping('val_early_stop_on', min_delta=.000, verbose=True, mode="min", patience=10)]
        self.callbacks.append(LearningRateLogger())
        #self.callbacks.append(LoggingCallback())

    def set_args(self, args):
        # args["accumulate_grad_batches"] = {5: 2, 20: 3}
        return args


class LoggingCallback(Callback):

    def on_sanity_check_end(self, trainer, pl_module):
        pl_module.logger.log_graph(pl_module.model)
