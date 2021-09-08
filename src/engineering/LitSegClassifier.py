from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import ConfusionMatrix

from src.engineering.LitBase import LitBase
from src.engineering.PSDDataModule import *
from torch.nn import LogSoftmax
from torch import argmax

from src.evaluation.PIDEvaluator import PIDEvaluator
import logging


class LitSegClassifier(LitBase):

    def __init__(self, config, trial=None):
        super(LitSegClassifier, self).__init__(config, trial)
        self.n_type = config.system_config.n_type
        self.softmax = LogSoftmax(dim=1)
        self.accuracy = Accuracy()
        self.confusion = ConfusionMatrix(num_classes=self.n_type)
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
        self.evaluator = PIDEvaluator(self.logger, **eval_params)

    def _process_batch(self, batch):
        additional_fields = None
        (c, f), target = batch
        if isinstance(f, list):
            additional_fields = f[1:]
            f = f[0]
        if self.write_onnx:
            self.write_model([c, f])
        if self.occlude_index:
            f[:, self.occlude_index] = 0
        predictions = self.model([c, f])
        loss = self.criterion.forward(predictions, target)
        #loss, target_tensor, predictions, _ = self._calc_segment_loss( c, predictions, target)
        return loss, predictions, target, c, f, additional_fields

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _, _ = self._process_batch(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions, target, _, _, _ = self._process_batch(batch)
        pred = argmax(self.softmax(predictions), dim=1)
        acc = self.accuracy(pred, target)
        results_dict = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(results_dict, on_epoch=True, prog_bar=True, logger=True)
        if self.pylog.level <= logging.INFO:
            if not hasattr(self, "confusion_matrix"):
                self.confusion_matrix = self.confusion(pred, target)
            else:
                self.confusion_matrix += self.confusion(pred, target)
        return results_dict

    def test_step(self, batch, batch_idx):
        loss, predictions, target, c, f, additional_fields = self._process_batch(batch)
        pred = argmax(self.softmax(predictions), dim=1)
        acc = self.accuracy(pred, target)
        results_dict = {'test_loss': loss, 'test_acc': acc}
        if not hasattr(self, "test_confusion_matrix"):
            self.test_confusion_matrix = self.confusion(pred, target)
        else:
            self.test_confusion_matrix += self.confusion(pred, target)
        if not self.evaluator.logger:
            self.evaluator.logger = self.logger
        self.evaluator.add(pred, target, c, additional_fields)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict

