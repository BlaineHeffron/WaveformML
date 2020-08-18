from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger



class PSDCallbacks:

    def __init__(self, config):
        self.config = config
        self.callbacks = [EarlyStopping('val_loss', min_delta=.01, verbose=True)]
        self.callbacks.append(LearningRateLogger())

    def set_args(self, args):
        #args["accumulate_grad_batches"] = {5: 2, 20: 3}
        return args

