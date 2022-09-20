from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import ConfusionMatrix
from torch_geometric.data import Data

from src.engineering.LitBase import LitBase
from src.engineering.PSDDataModule import *
from torch.nn import Softmax
from torch import argmax, cat, unsqueeze

from src.evaluation.PIDEvaluator import PIDEvaluator
from src.evaluation.ROCCurve import ROCCurve
import logging


class LitSegClassifier(LitBase):

    def __init__(self, config, trial=None):
        super(LitSegClassifier, self).__init__(config, trial)
        self.n_type = config.system_config.n_type
        self.softmax = Softmax(dim=1)
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
        self.roc_curve = ROCCurve(class_name=self.evaluator.class_names[0])

    def _process_batch(self, batch):
        additional_fields = None
        if isinstance(batch, Data):
            if hasattr(batch, "additional_fields"):
                additional_fields = batch.additional_fields
            if self.write_script:
                self.write_model(batch)
            if self.occlude_index:
                batch.x[:, self.occlude_index] = 0
            predictions = self.model(batch).squeeze(1)
            c = cat([batch.pos, unsqueeze(batch.batch, dim=1)], dim=1)
            target = batch.y
            f = batch.x
        else:
            (c, f), target = batch
            if isinstance(f, list):
                additional_fields = f[1:]
                f = f[0]
            if self.write_script:
                self.write_model([c, f])
            if self.occlude_index:
                f[:, self.occlude_index] = 0
            predictions = self.model([c, f])
        if self.SE_only:
            se_inds = self.SE_mask[0, 0, c[:, 0].long(), c[:, 1].long()] == 1.0
            loss = self.criterion.forward(predictions[se_inds], target[se_inds])
        else:
            loss = self.criterion.forward(predictions, target)
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
        prob = self.softmax(predictions)
        pred = argmax(prob, dim=1)
        acc = self.accuracy(pred, target)
        results_dict = {'test_loss': loss, 'test_acc': acc}
        if not hasattr(self, "test_confusion_matrix"):
            self.test_confusion_matrix = self.confusion(pred, target)
        else:
            self.test_confusion_matrix += self.confusion(pred, target)
        self.roc_curve.update(prob, target)
        if not self.evaluator.logger:
            self.evaluator.logger = self.logger
        self.evaluator.add(pred, target, c, additional_fields)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict

