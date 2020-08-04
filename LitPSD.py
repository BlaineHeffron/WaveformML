import pytorch_lightning as pl
from PSDDataModule import *
import torch


class LitPSD(pl.LightningModule):

    def __init__(self, config):
        super(LitPSD, self).__init__()
        self.config = config
        self.n_type = config.system_config.n_type
        self.modules = ModuleUtility(config.net_config.imports + config.dataset_config.imports +
                config.optimize_config.imports)
        self.model_class = self.modules.retrieve_class(config.net_config.net_class)
        self.model = self.model_class(config)
        self.data_module = PSDDataModule(config.dataset_config)
        self.criterion_class = self.modules.retrieve_class(config.net_config.criterion_class)
        self.criterion = self.criterion_class(*config.net_config.criterion_params)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def prepare_data(self):
        self.data_module.prepare_data()

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def configure_optimizers(self):
        optimizer = \
            self.modules.retrieve_class(self.config.optimize_config.optimizer_class)(self.model.parameters(),
                                                                                     **DictionaryUtility.to_dict(
                                                                                         self.config.optimize_config.optimizer_params))
        if hasattr(self.config.optimize_config, "lr_scheduler_class"):
            if self.config.optimize_config.lr_scheduler_class:
                if not hasattr(self.config.optimize_config, "lr_scheduler_parameters"):
                    raise IOError(
                        "Optimizer config has a learning scheduler class specified. You must also set lr_schedule_parameters (dictionary of key value pairs).")
                scheduler = self.modules.retrieve_class(self.config.optimize_config.lr_scheduler_class)(
                    **DictionaryUtility.to_dict(self.config.optimize_config.lr_scheduler_parameters))
                return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        predictions = self.model(batch[0])
        loss = self.criterion.forward(predictions, batch[1])
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        predictions = self.model(batch[0])
        loss = self.criterion.forward(predictions, batch[1])
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        result.log('predictions',predictions)
        result.log('predictions',predictions)
        return result

    def validation_epoch_end(self, outputs):
        mean_loss = outputs['val_loss'].mean()
        return {
            'log': {'mean_loss': mean_loss},
            'progress_bar': {'mean_loss': mean_loss}
        }


