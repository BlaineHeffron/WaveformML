from src.engineering.LitBase import LitBase
from src.engineering.PSDDataModule import *
from torchmetrics import MeanSquaredError

from src.evaluation.SegEvaluator import SegEvaluator


class LitSegQuantifier(LitBase):

    def __init__(self, config, trial=None):
        super(LitSegQuantifier, self).__init__(config, trial)
        self.MSE = MeanSquaredError()
        if hasattr(self.config.dataset_config, "calgroup"):
            calgroup = self.config.dataset_config.calgroup
        else:
            calgroup = None
        eval_params = {"calgroup": calgroup}
        if hasattr(config.dataset_config, "test_dataset_params"):
            if hasattr(config.dataset_config.test_dataset_params, "additional_fields"):
                eval_params["additional_field_names"] = config.dataset_config.test_dataset_params.additional_fields
        if hasattr(config, "evaluation_config"):
            eval_params = DictionaryUtility.to_dict(config.evaluation_config)
        self.target_index = config.dataset_config.dataset_params.label_index
        self.evaluator = SegEvaluator(self.logger, **eval_params)

    def _process_batch(self, batch):
        additional_fields = None
        (c, f), target = batch
        use_target_index = len(target.shape) > 1
        if isinstance(f, list):
            additional_fields = f[1:]
            f = f[0]
        if self.write_onnx:
            self.write_model([c, f])
        if self.occlude_index:
            f[:, self.occlude_index] = 0
        predictions = self.model([c, f]).squeeze(1)
        if self.SE_only:
            se_inds = self.SE_mask[0, 0, c[:, 0].long(), c[:, 1].long()] == 1.0
            if use_target_index:
                loss = self.criterion.forward(predictions[se_inds], target[se_inds, self.target_index])
                mse = self.MSE(predictions[se_inds], target[se_inds, self.target_index])
            else:
                loss = self.criterion.forward(predictions[se_inds], target[se_inds])
                mse = self.MSE(predictions[se_inds], target[se_inds])
        else:
            if use_target_index:
                loss = self.criterion.forward(predictions, target[:, self.target_index])
                mse = self.MSE(predictions, target[:, self.target_index])
            else:
                loss = self.criterion.forward(predictions, target)
                mse = self.MSE(predictions, target)
        return loss, predictions, target, c, f, additional_fields, mse

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _, _, _ = self._process_batch(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions, target, _, _, _, mse = self._process_batch(batch)
        results_dict = {'val_loss': loss, 'val_mse': mse}
        self.log_dict(results_dict, on_epoch=True, prog_bar=True, logger=True)
        return results_dict

    def test_step(self, batch, batch_idx):
        loss, predictions, target, c, f, additional_fields, mse = self._process_batch(batch)
        results_dict = {'test_loss': loss, 'test_mse': mse}
        if not self.evaluator.logger:
            self.evaluator.logger = self.logger
        self.evaluator.add(predictions, target, c, additional_fields)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict

