from pytorch_lightning.metrics import Accuracy, ConfusionMatrix
from torch import argmax, cat
from torch.nn import Softmax

from src.engineering.LitBase import LitBase
from src.evaluation.TensorEvaluator import TensorEvaluator


class LitWaveform(LitBase):

    def __init__(self, config, trial=None):
        if hasattr(config.net_config, "use_detector_number"):
            self.use_detector_number = config.net_config.use_detector_number
            if self.use_detector_number:
                if not hasattr(config.net_config, "num_detectors"):
                    raise IOError(
                        "net config must contain 'num_detectors' property if 'use_detector_number' set to true")
                config.system_config.n_samples = config.system_config.n_samples + 1
                self.detector_num_factor = 1. / config.net_config.num_detectors
        else:
            self.use_detector_number = False
        super(LitWaveform, self).__init__(config, trial)
        if config.net_config.net_class.endswith("RecurrentWaveformNet"):
            self.squeeze_index = 2
        else:
            self.squeeze_index = 1
        self.test_has_phys = False
        if hasattr(self.config.dataset_config, "test_dataset_params"):
            if self.config.dataset_config.test_dataset_params.label_name == "phys" and not hasattr(
                    self.config.dataset_config.test_dataset_params, "label_index"):
                self.test_has_phys = True
        if hasattr(self.config.dataset_config, "calgroup"):
            calgroup = self.config.dataset_config.calgroup
        else:
            calgroup = None
        if hasattr(self.config.dataset_config.dataset_params, "label_index"):
            self.target_index = self.config.dataset_config.dataset_params.label_index
        else:
            self.target_index = None
        self.use_accuracy = False
        if config.net_config.criterion_class == "L1Loss":
            metric_name = "mean absolute error"
        elif config.net_config.criterion_class == "MSELoss":
            metric_name = "mean squared error"
        elif config.net_config.criterion_class.startswith("BCE") or config.net_config.criterion_class.startswith(
                "CrossEntropy"):
            self.use_accuracy = True
            metric_name = "Accuracy"
        else:
            metric_name = "?"
        self.evaluator = TensorEvaluator(self.logger, calgroup=calgroup,
                                         target_has_phys=self.test_has_phys, target_index=self.target_index,
                                         metric_name=metric_name)
        self.loss_no_reduce = self.criterion_class(*config.net_config.criterion_params, reduction="none")
        if self.use_accuracy:
            self.accuracy = Accuracy()
            self.confusion = ConfusionMatrix(2)
            self.softmax = Softmax(dim=1)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (c, f), target = batch
        if self.use_detector_number:
            f = cat((f, (c * self.detector_num_factor).unsqueeze(1)), dim=1)
        predictions = self.model(f.unsqueeze(self.squeeze_index)).squeeze(1)
        loss = self.criterion.forward(predictions, target)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (c, f), target = batch
        if self.use_detector_number:
            f = cat((f, (c * self.detector_num_factor).unsqueeze(1)), dim=1)
        predictions = self.model(f.unsqueeze(self.squeeze_index)).squeeze(1)
        loss = self.criterion.forward(predictions, target)
        results_dict = {'val_loss': loss}
        if self.use_accuracy:
            pred = argmax(self.softmax(predictions), dim=1)
            acc = self.accuracy(pred, target)
            results_dict["val_accuracy"] = acc
        self.log_dict(results_dict, on_epoch=True, prog_bar=True, logger=True)
        return results_dict

    def test_step(self, batch, batch_idx):
        (c, f), target = batch
        if self.use_detector_number:
            f = cat((f, (c * self.detector_num_factor).unsqueeze(1)), dim=1)
        predictions = self.model(f.unsqueeze(self.squeeze_index)).squeeze(1)
        if self.test_has_phys:
            loss = self.criterion.forward(predictions, target[:, self.target_index])
        else:
            loss = self.criterion.forward(predictions, target)
        results_dict = {'test_loss': loss}
        if self.use_accuracy:
            pred = argmax(self.softmax(predictions), dim=1)
            acc = self.accuracy(pred, target)
            results_dict["val_accuracy"] = acc
            if not hasattr(self, "test_confusion_matrix"):
                self.test_confusion_matrix = self.confusion(pred, target)
            else:
                self.test_confusion_matrix += self.confusion(pred, target)
        if hasattr(self, "evaluator"):
            if not self.evaluator.logger:
                self.evaluator.logger = self.logger
            if self.test_has_phys:
                results = self.loss_no_reduce(predictions, target[:, self.target_index])
            else:
                results = self.loss_no_reduce(predictions, target)
            self.evaluator.add(target, results)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict

