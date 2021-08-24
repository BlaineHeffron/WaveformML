from src.engineering.LitBase import LitBase
from src.evaluation.EZEvaluator import EZEvaluatorWF, EZEvaluatorPhys
from src.models.SingleEndedEZConv import SingleEndedEZConv
from src.engineering.PSDDataModule import *


class LitEZ(LitBase):

    def __init__(self, config, trial=None):
        super(LitEZ, self).__init__(config, trial, event_predictions=False)
        self.model = SingleEndedEZConv(self.config)
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
                self.evaluator = EZEvaluatorPhys(self.logger, calgroup=self.config.dataset_config.calgroup,
                                                 e_scale=self.e_adjust)
            else:
                self.evaluator = EZEvaluatorPhys(self.logger, e_scale=self.e_adjust)
        else:
            if hasattr(self.config.dataset_config, "calgroup"):
                self.evaluator = EZEvaluatorWF(self.logger, calgroup=self.config.dataset_config.calgroup,
                                               e_scale=self.e_adjust)
            else:
                self.evaluator = EZEvaluatorWF(self.logger, e_scale=self.e_adjust)

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

    def _process_batch(self, batch):
        (c, f), target = batch
        if self.phys_coord and self.e_factor != 1.:
            f[:, 0] *= self.e_factor
            f[:, 2] *= self.e_factor
            f[:, 3] *= self.e_factor
        if self.occlude_index:
            f[:, self.occlude_index] = 0
        if self.write_onnx:
            self.write_model([c, f])
        predictions = self.model([c, f])
        ZLoss, target_tensor_z, predictions_z, sparse_mask = self._calc_segment_loss(c, predictions[:, 0].unsqueeze(1), target[:, 0])
        ELoss, target_tensor_E, predictions_E, _ = self._calc_segment_loss(c, predictions[:, 1].unsqueeze(1), target[:, 1], sparse_mask=sparse_mask)
        predictions = cat((predictions_z, predictions_E), dim=1)
        target_tensor = cat((target_tensor_z, target_tensor_E), dim=1)
        loss = ZLoss + ELoss
        return c, f, predictions, target_tensor, loss, ELoss, ZLoss

    def training_step(self, batch, batch_idx):
        _, _, predictions, target_tensor, loss, ELoss, ZLoss = self._process_batch(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, predictions, target_tensor, loss, ELoss, ZLoss = self._process_batch(batch)
        results_dict = {'val_loss': loss, 'val_MAE_E': ELoss, 'val_MAE_z': ZLoss}
        self.log_dict(results_dict, on_epoch=True, prog_bar=True, logger=True)
        return results_dict

    def test_step(self, batch, batch_idx):
        c, f, predictions, target_tensor, loss, ELoss, ZLoss = self._process_batch(batch)
        results_dict = {'test_loss': loss, 'test_MAE_E': ELoss, 'test_MAE_z': ZLoss}
        if not self.evaluator.logger:
            self.evaluator.set_logger(self.logger)
        self.evaluator.add(predictions, target_tensor, c, f)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict
