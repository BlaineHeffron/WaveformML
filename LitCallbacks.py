from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger
from os.path import join



class PSDCallbacks:

    def __init__(self, config):
        self.callbacks = {"callbacks": []}
        self.set_callbacks()

    def set_callbacks(self):
        self.callbacks["early_stop_callback"] = EarlyStopping('val_loss')
        self.callbacks["accumulate_grad_batches"] = {5: 2, 20: 3}
        self.callbacks["callbacks"].append(LearningRateLogger)
