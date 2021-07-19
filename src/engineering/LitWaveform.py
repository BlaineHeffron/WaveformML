from pytorch_lightning.metrics import Accuracy, ConfusionMatrix
from torch import argmax, cat, floor_divide, zeros, float32, int32
from torch.nn import Softmax

from src.engineering.LitBase import LitBase
from src.evaluation.TensorEvaluator import TensorEvaluator
from src.utils.util import DictionaryUtility


class LitWaveform(LitBase):

    def __init__(self, config, trial=None):
        if hasattr(config.net_config, "use_detector_number"):
            self.use_detector_number = config.net_config.use_detector_number
            if self.use_detector_number:
                if not hasattr(config.net_config, "num_detectors"):
                    raise IOError(
                        "net config must contain 'num_detectors' property if 'use_detector_number' set to true")
                config.system_config.n_samples = config.system_config.n_samples + 3
                if config.net_config.num_detectors == 308:
                    self.detector_num_factor_x = 1. / 13
                    self.detector_num_factor_y = 1. / 10
                else:
                    raise IOError("num detectors " + str(config.net_config.num_detector) + " not supported")
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
        eval_params = {}
        if hasattr(config, "evaluation_config"):
            eval_params = DictionaryUtility.to_dict(config.evaluation_config)
        self.evaluator = TensorEvaluator(self.logger, calgroup=calgroup,
                                         target_has_phys=self.test_has_phys, target_index=self.target_index,
                                         metric_name=metric_name, **eval_params)
        self.loss_no_reduce = self.criterion_class(*config.net_config.criterion_params, reduction="none")
        if self.use_accuracy:
            self.accuracy = Accuracy()
            self.confusion = ConfusionMatrix(2)
            self.softmax = Softmax(dim=1)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def fill_coords(self, coords, det):
        seg = floor_divide(det, 2)
        coords[:, 0] = (seg % 14) * self.detector_num_factor_x
        coords[:, 1] = floor_divide(seg, 14) * self.detector_num_factor_y
        coords[:, 2] = det % 2

    def training_step(self, batch, batch_idx):
        (c, f), target = batch
        if self.use_detector_number:
            coords = zeros((f.shape[0], 3), dtype=f.dtype, device=f.device)
            self.fill_coords(coords, c)
            f = cat((f, coords), dim=1)
            # f = cat((f, ((c % 14) * self.detector_num_factor_x).unsqueeze(1),
            #         (floor_divide(c, 14) * self.detector_num_factor_y).unsqueeze(1), (c%2).unsqueeze(1)), dim=1)
        predictions = self.model(f.unsqueeze(self.squeeze_index)).squeeze(1)
        if predictions.dim() == 2 and target.dim() == 1:
            predictions = predictions.squeeze(1)
        loss = self.criterion.forward(predictions, target)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (c, f), target = batch
        if self.use_detector_number:
            coords = zeros((f.shape[0], 3), dtype=f.dtype, device=f.device)
            self.fill_coords(coords, c)
            f = cat((f, coords), dim=1)
            # f = cat((f, ((c % 14) * self.detector_num_factor_x).unsqueeze(1),
            #         (floor_divide(c, 14) * self.detector_num_factor_y).unsqueeze(1), (c%2).unsqueeze(1)), dim=1)
        predictions = self.model(f.unsqueeze(self.squeeze_index)).squeeze(1)
        if predictions.dim() == 2 and target.dim() == 1:
            predictions = predictions.squeeze(1)
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
            coords = zeros((f.shape[0], 3), dtype=f.dtype, device=f.device)
            self.fill_coords(coords, c)
            f = cat((f, coords), dim=1)
            # f = cat((f, ((c % 14) * self.detector_num_factor_x).unsqueeze(1),
            #         (floor_divide(c, 14) * self.detector_num_factor_y).unsqueeze(1), (c%2).unsqueeze(1)), dim=1)
        predictions = self.model(f.unsqueeze(self.squeeze_index)).squeeze(1)
        if predictions.dim() == 2 and (target.dim() == 1 or (target.dim() == 2 and self.test_has_phys)):
            predictions = predictions.squeeze(1)
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
