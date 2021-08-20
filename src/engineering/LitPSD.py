from pytorch_lightning.metrics.classification import Accuracy
from torchmetrics import ConfusionMatrix

from src.engineering.LitBase import LitBase
from src.engineering.PSDDataModule import *
from torch.nn import LogSoftmax
from torch import argmax, sum
from src.evaluation.PSDEvaluator import PSDEvaluator, PhysEvaluator
import logging

from src.evaluation.TensorEvaluator import TensorEvaluator

N_CHANNELS = 14


def weight_avg(t, n):
    return sum(t * n / sum(n))


class LitPSD(LitBase):

    def __init__(self, config, trial=None):
        super(LitPSD, self).__init__(config, trial)
        self.n_type = config.system_config.n_type
        self.softmax = LogSoftmax(dim=1)
        self.accuracy = Accuracy()
        self.confusion = ConfusionMatrix(num_classes=self.n_type)
        if hasattr(self.config.dataset_config, "calgroup"):
            calgroup = self.config.dataset_config.calgroup
        else:
            calgroup = None
        eval_params = {}
        if hasattr(config, "evaluation_config"):
            eval_params = DictionaryUtility.to_dict(config.evaluation_config)
        if self.config.dataset_config.dataset_class == "PulseDatasetDet":
            self.evaluator = PhysEvaluator(self.config.system_config.type_names, self.logger, device=self.device, **eval_params)
        elif self.config.dataset_config.dataset_class == "PulseDatasetWaveformNorm":
            self.evaluator = TensorEvaluator(self.logger, calgroup=calgroup,
                                             target_has_phys=False, target_index=None,
                                             metric_name="accuracy", metric_unit="", **eval_params)
        else:
            if hasattr(self.config.dataset_config, "calgroup"):
                self.evaluator = PSDEvaluator(self.config.system_config.type_names, self.logger, device=self.device,
                                              calgroup=self.config.dataset_config.calgroup, **eval_params)
            else:
                self.evaluator = PSDEvaluator(self.config.system_config.type_names, self.logger, device=self.device, **eval_params)

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

    """
    def convert_to_tensors(self, coord, feat, label):
        if self.needs_float:
            if self.on_gpu:
                feat = feat.type(float32).cuda(self.device)
            else:
                feat = feat.type(float32)
            feat /= 2**N_CHANNELS-1
        if self.on_gpu:
            label = label.type(int64).cuda(self.device)
            coord = coord.cuda(self.device)
        else:
            label = label.type(int64)
        return coord, feat, label
    """

    def training_step(self, batch, batch_idx):
        (c, f), target = batch
        # c, f, target = self.convert_to_tensors(c, f, target)
        # self.log.debug("type of coords: {}".format(c.storage_type()))
        # self.log.debug("type of features: {}".format(f.storage_type()))
        # self.log.debug("type of labels: {}".format(target.storage_type()))
        predictions = self.model([c, f])
        # self.log.debug("predictions shape is {}".format(predictions.shape))
        loss = self.criterion.forward(predictions, target)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (c, f), target = batch
        # c, f, target = self.convert_to_tensors(c, f, target)
        predictions = self.model([c, f])
        loss = self.criterion.forward(predictions, target)
        pred = argmax(self.softmax(predictions), dim=1)
        acc = self.accuracy(pred, target)
        results_dict = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(results_dict, on_epoch=True, prog_bar=True, logger=True)
        if self.pylog.level <= logging.INFO:
            if not hasattr(self, "confusion_matrix"):
                self.confusion_matrix = self.confusion(pred, target)
            else:
                self.confusion_matrix += self.confusion(pred, target)

        """
        if self.trial:
            self.trial.report(loss.detach().item(), batch_idx)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        """

        return results_dict

    def test_step(self, batch, batch_idx):
        (c, f), target = batch
        if self.write_onnx:
            self.write_model([c,f])
        if self.occlude_index:
            f[:, self.occlude_index] = 0
        predictions = self.model([c, f])
        loss = self.criterion.forward(predictions, target)
        pred = argmax(self.softmax(predictions), dim=1)
        #if batch_idx == 0:
        #    self.logger.experiment.add_graph(self.model, [c, f])
        acc = self.accuracy(pred, target)
        results_dict = {'test_loss': loss, 'test_acc': acc}
        if not hasattr(self, "test_confusion_matrix"):
            self.test_confusion_matrix = self.confusion(pred, target)
        else:
            self.test_confusion_matrix += self.confusion(pred, target)
        if not self.evaluator.logger:
            self.evaluator.logger = self.logger
        self.evaluator.add(batch, predictions, pred)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict

