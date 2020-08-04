from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger
from os.path import join



class PSDCallbacks:

    def __init__(self, config):
        self.config = config

    def set_args(self, args):
        args["early_stop_callback"] = EarlyStopping('val_loss')
        args["accumulate_grad_batches"] = {5: 2, 20: 3}
        args["callbacks"].append(LearningRateLogger)
        return args

