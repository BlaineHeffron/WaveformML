import spconv

from src.evaluation.EZEvaluator import EZEvaluatorWF, EZEvaluatorPhys
from src.models.SingleEndedEZConv import SingleEndedEZConv
from src.engineering.PSDDataModule import *
from torch import where


class LitEZ(pl.LightningModule):

    def __init__(self, config, trial=None):
        super(LitEZ, self).__init__()
        if trial:
            self.trial = trial
        else:
            self.trial = None
        self.config = config
        if hasattr(config.system_config, "half_precision"):
            self.needs_float = not config.system_config.half_precision
        else:
            self.needs_float = True
        self.hparams = DictionaryUtility.to_dict(config)
        self.lr = config.optimize_config.lr
        self.modules = ModuleUtility(config.net_config.imports + config.dataset_config.imports +
                                     config.optimize_config.imports)
        self.model = SingleEndedEZConv(self.config)
        self.criterion_class = self.modules.retrieve_class(config.net_config.criterion_class)
        self.criterion = self.criterion_class(*config.net_config.criterion_params)
        self.zscale = 1200.
        self.escale = 12.
        self.e_adjust = 12.
        self.phys_coord = False
        if hasattr(config.net_config, "escale"):
            self.escale = config.net_config.escale
        if hasattr(config.net_config, "zscale"):
            self.zscale = config.net_config.zscale
        if hasattr(config.net_config, "e_adjust"):
            self.e_adjust = config.net_config.e_adjust
        self.e_factor = self.escale / self.e_adjust
        if config.net_config.algorithm == "features":
            self.phys_coord = True
            if hasattr(self.config.dataset_config, "calgroup"):
                self.evaluator = EZEvaluatorPhys(self.logger, calgroup=self.config.dataset_config.calgroup, e_scale=self.e_adjust)
            else:
                self.evaluator = EZEvaluatorPhys(self.logger, e_scale=self.e_adjust)
        else:
            if hasattr(self.config.dataset_config, "calgroup"):
                self.evaluator = EZEvaluatorWF(self.logger, calgroup=self.config.dataset_config.calgroup, e_scale=self.e_adjust)
            else:
                self.evaluator = EZEvaluatorWF(self.logger, e_scale=self.e_adjust)

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

    def _format_target_and_prediction(self, pred, coords, target, batch_size):
        if self.e_factor != 1.:
            target[:, 0] *= self.e_factor
        target_tensor = spconv.SparseConvTensor(target, coords[:, self.model.permute_tensor],
                                                self.model.spatial_size, batch_size)
        target_tensor = target_tensor.dense()
        # set output to 0 if there was no value for input
        return where(target_tensor == 0, target_tensor, pred), target_tensor

    def _calc_loss(self, p, t):
        ELoss = self.criterion.forward(p[:, 0, :, :], t[:, 0, :, :])
        ZLoss = self.criterion.forward(p[:, 1, :, :], t[:, 1, :, :])
        return ELoss + ZLoss, self.escale*ELoss, self.zscale*ZLoss

    def _process_batch(self, batch):
        (c, f), target = batch
        if self.phys_coord and self.e_factor != 1.:
            f[:, 0] *= self.e_factor
            f[:, 2] *= self.e_factor
            f[:, 3] *= self.e_factor
        predictions = self.model([c, f])
        batch_size = c[-1, -1] + 1
        predictions, target_tensor = self._format_target_and_prediction(predictions, c, target, batch_size)
        return c, f, predictions, target_tensor


    def training_step(self, batch, batch_idx):
        _, _, predictions, target_tensor = self._process_batch(batch)
        loss, ELoss, ZLoss = self._calc_loss(predictions, target_tensor)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, predictions, target_tensor = self._process_batch(batch)
        loss, ELoss, ZLoss = self._calc_loss(predictions, target_tensor)
        results_dict = {'val_loss': loss, 'val_MAE_E': ELoss, 'val_MAE_z': ZLoss}
        self.log_dict(results_dict, on_epoch=True, prog_bar=True, logger=True)
        return results_dict

    def test_step(self, batch, batch_idx):
        c, f, predictions, target_tensor = self._process_batch(batch)
        loss, ELoss, ZLoss = self._calc_loss(predictions, target_tensor)
        results_dict = {'test_loss': loss, 'test_MAE_E': ELoss, 'test_MAE_z': ZLoss}
        if not self.evaluator.logger:
            self.evaluator.set_logger(self.logger)
        self.evaluator.add(predictions, target_tensor, c, f)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict
