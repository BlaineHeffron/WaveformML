from pytorch_lightning.metrics.functional import accuracy
from src.engineering.PSDDataModule import *
from torch.nn import Softmax
from torch import argmax, max, int32, float32, int64, tensor
import logging


class LitPSD(pl.LightningModule):

    def __init__(self, config):
        super(LitPSD, self).__init__()
        self.log = logging.getLogger(__name__)
        logging.getLogger("lightning").setLevel(self.log.level)
        self.config = config
        self.hparams = DictionaryUtility.to_dict(config)
        self.n_type = config.system_config.n_type
        self.lr = config.optimize_config.lr
        self.modules = ModuleUtility(config.net_config.imports + config.dataset_config.imports +
                                     config.optimize_config.imports)
        self.model_class = self.modules.retrieve_class(config.net_config.net_class)
        self.model = self.model_class(config)
        self.criterion_class = self.modules.retrieve_class(config.net_config.criterion_class)
        self.criterion = self.criterion_class(*config.net_config.criterion_params)
        self.softmax = Softmax(dim=1)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    """
    def prepare_data(self):
        self.data_module.prepare_data()

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()
    """

    def configure_optimizers(self):
        optimizer = \
            self.modules.retrieve_class(self.config.optimize_config.optimizer_class)(self.model.parameters(),
                                                                                     lr=(self.lr or self.learning_rate),
                                                                                     **DictionaryUtility.to_dict(
                                                                                         self.config.optimize_config.optimizer_params))
        if hasattr(self.config.optimize_config, "scheduler_class"):
            if self.config.optimize_config.scheduler_class:
                if not hasattr(self.config.optimize_config, "scheduler_params"):
                    raise IOError(
                        "Optimizer config has a learning scheduler class specified. You must also set "
                        "lr_schedule_parameters (dictionary of key value pairs).")
                scheduler = self.modules.retrieve_class(self.config.optimize_config.scheduler_class)(optimizer,
                                                                                                     **DictionaryUtility.to_dict(
                                                                                                         self.config.optimize_config.scheduler_params))
                return [optimizer], [scheduler]
        return optimizer

    def convert_to_tensors(self, coord, feat, label):
        coord = tensor(coord, device=self.device, dtype=int32)
        feat = tensor(feat, device=self.device, dtype=float32)
        label = tensor(label, device=self.device, dtype=int64)
        return coord, feat, label

    def training_step(self, batch, batch_idx):
        (coo, feat), targets = batch
        self.log.debug("Shape of coords: {}".format(coo.shape))
        self.log.debug("Shape of features: {}".format(feat.shape))
        self.log.debug("Shape of labels: {}".format(targets.shape))
        for c, f, target in zip(coo, feat, targets):
            #c, f, target = self.convert_to_tensors(c, f, target)
            #self.log.debug("Shape of coords: {}".format(c.shape))
            #self.log.debug("Shape of features: {}".format(f.shape))
            #self.log.debug("Shape of labels: {}".format(target.shape))
            predictions = self.model([c, f])
            loss = self.criterion.forward(predictions, target)
            result = pl.TrainResult(loss)
            result.log('train_loss', loss)
            if self.log.level > 4:
                self.log.debug("batch id: {0}, train_loss: {1}"
                               .format(batch_idx, str(loss.item())))
        return result

    def validation_step(self, batch, batch_idx):
        (coo, feat), targets = batch
        for c, f, target in zip(coo, feat, targets):
            predictions = self.model([c, f])
            loss = self.criterion.forward(predictions, target)
            result = pl.EvalResult(checkpoint_on=loss)
            pred = argmax(self.softmax(predictions), dim=1)
            acc = accuracy(pred, target, num_classes=self.n_type)
            result.log_dict({'val_loss': loss, 'val_acc': acc}, on_epoch=True)
            if self.log.level > 4:
                self.log.debug("batch id: {0}, val_loss: {1}, val_acc: {2}"
                               .format(batch_idx, str(loss.item()), str(acc.item())))
        return result

    """
    def validation_epoch_end(self, outputs):
        mean_loss = outputs['batch_val_loss'].mean()
        mean_accuracy = outputs['batch_val_acc'].mean()
        return {
            'log': {'val_loss': mean_loss, 'val_acc': mean_accuracy},
            'progress_bar': {'val_loss': mean_loss},
            'callback_metrics': {'val_loss': mean_loss, 'val_acc': mean_accuracy}
        }
    """
