from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger
from os.path import join



class PSDCallbacks:

    def __init__(self, config):
        self.config = config
        self.callbacks = []
        self.callbacks.append(LearningRateLogger)

    def set_args(self, args):
        args.early_stop_callback = EarlyStopping('val_loss')
        #args.accumulate_grad_batches = {5: 2, 20: 3}
        return args

