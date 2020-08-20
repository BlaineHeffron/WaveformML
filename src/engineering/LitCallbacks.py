from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger



class PSDCallbacks:

    def __init__(self, config):
        self.config = config
        #self.callbacks = [EarlyStopping('val_checkpoint_on', min_delta=.001, verbose=True, mode="min", patience=7)]
        self.callbacks.append(LearningRateLogger())

    def set_args(self, args):
        #args["accumulate_grad_batches"] = {5: 2, 20: 3}
        return args

