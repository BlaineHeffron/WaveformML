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
        if hasattr(config.system_config,"half_precision"):
            self.needs_float = not config.system_config.half_precision
        else:
            self.needs_float = True
        self.hparams = DictionaryUtility.to_dict(config)
        self.n_type = config.system_config.n_type
        self.lr = config.optimize_config.lr
        self.modules = ModuleUtility(config.net_config.imports + config.dataset_config.imports +
                                     config.optimize_config.imports)
        self.model_class = self.modules.retrieve_class(config.net_config.net_class)
        #self.data_module = PSDDataModule(config,self.device)
        self.model = self.model_class(config)
        self.criterion_class = self.modules.retrieve_class(config.net_config.criterion_class)
        self.criterion = self.criterion_class(*config.net_config.criterion_params)
        self.softmax = Softmax(dim=1)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    """
    def prepare_data(self):
        self.data_module.prepare_data()
        self.data_module.setup()

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
        if self.needs_float:
            if self.on_gpu:
                feat = feat.type(float32).cuda(self.device)
            else:
                feat = feat.type(float32)
        if self.on_gpu:
            label = label.type(int64).cuda(self.device)
            coord = coord.cuda(self.device)
        else:
            label = label.type(int64)
        return coord, feat, label

    def training_step(self, batch, batch_idx):
        (coo, feat), targets = batch
        #self.log.debug("Shape of coords: {}".format(coo.shape))
        #self.log.debug("Shape of features: {}".format(feat.shape))
        #self.log.debug("Shape of labels: {}".format(targets.shape))
        for c, f, target in zip(coo, feat, targets):
            c, f, target = self.convert_to_tensors(c, f, target)
            #self.log.debug("type of coords: {}".format(c.storage_type()))
            #self.log.debug("type of features: {}".format(f.storage_type()))
            #self.log.debug("type of labels: {}".format(target.storage_type()))
            predictions = self.model([c, f])
            #self.log.debug("predictions shape is {}".format(predictions.shape))
            loss = self.criterion.forward(predictions, target)
            result = pl.TrainResult(loss)
            result.log('train_loss', loss)
            if self.log.level == logging.DEBUG:
                self.log.debug("batch id: {0}, train_loss: {1}"
                               .format(batch_idx, str(loss.item())))
        return result

    def validation_step(self, batch, batch_idx):
        (coo, feat), targets = batch
        for c, f, target in zip(coo, feat, targets):
            c, f, target = self.convert_to_tensors(c, f, target)
            predictions = self.model([c, f])
            loss = self.criterion.forward(predictions, target)
            result = pl.EvalResult(checkpoint_on=loss)
            pred = argmax(self.softmax(predictions), dim=1)
            acc = accuracy(pred, target, num_classes=self.n_type)
            result.log_dict({'val_loss': loss, 'val_acc': acc}, on_epoch=True)
            if self.log.level == logging.DEBUG:
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
