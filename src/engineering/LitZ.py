import spconv

from src.engineering.LitBase import LitBase
from src.models.SingleEndedZConv import SingleEndedZConv
from src.engineering.PSDDataModule import *
from torch import where, tensor, sum, cat, zeros, int32, floor_divide
from torch.fft import rfft
from src.evaluation.ZEvaluator import ZEvaluatorWF, ZEvaluatorPhys, ZEvaluatorRealWFNorm


def create_coord_from_det(c, f):
    coord = zeros((f.shape[0], 3), dtype=int32)
    seg = zeros(c.shape, dtype=c.dtype)
    floor_divide(c, 2, out=seg)
    coord[:, 0] = seg % 14
    coord[:,1] = floor_divide(seg, 14)
    features = zeros((f.shape[0],f.shape[1]*2), dtype=f.dtype)
    n_samp = f.shape[1]
    for i in range(coord.shape[0]):
        coord[i, 2] = i
        if c[i] % 2 == 0:
            features[i,0:n_samp] = f[i]
        else:
            features[i,n_samp:] = f[i]

    return coord, features


class LitZ(LitBase):

    def __init__(self, config, trial=None):
        super(LitZ, self).__init__(config, trial, event_predictions=False)
        self.model = SingleEndedZConv(self.config)
        if hasattr(self.config.dataset_config, "test_dataset_params"):
            if self.config.dataset_config.test_dataset_params.label_name == "phys" and not hasattr(
                    self.config.dataset_config.test_dataset_params, "label_index"):
                self.test_has_phys = True
        if hasattr(self.config.net_config, "UseFFT"):
            print("Using FFT")
            self.use_fft = True
        else:
            self.use_fft = False
        eval_params = {}
        if hasattr(config, "evaluation_config"):
            eval_params = DictionaryUtility.to_dict(config.evaluation_config)
        if hasattr(config.dataset_config, "test_dataset_params"):
            if hasattr(config.dataset_config.test_dataset_params, "additional_fields"):
                eval_params["additional_field_names"] = config.dataset_config.test_dataset_params.additional_fields
        if self.test_has_phys:
            if hasattr(self.config.dataset_config, "calgroup"):
                self.evaluator = ZEvaluatorRealWFNorm(self.logger, calgroup=self.config.dataset_config.calgroup, **eval_params)
            else:
                self.evaluator = ZEvaluatorRealWFNorm(self.logger, **eval_params)
        elif config.net_config.algorithm == "features":
            self.evaluator = ZEvaluatorPhys(self.logger, **eval_params)
        else:
            if hasattr(self.config.dataset_config, "calgroup"):
                self.evaluator = ZEvaluatorWF(self.logger, calgroup=self.config.dataset_config.calgroup)
            else:
                self.evaluator = ZEvaluatorWF(self.logger, **eval_params)
        if self.config.dataset_config.dataset_class == "PulseDatasetRealWFPair":
            self.target_is_cal = True
        else:
            self.target_is_cal = False
        if self.config.dataset_config.dataset_class == "PulseDatasetWaveformNorm":
            self.target_is_cal = True
            self.test_waveform = True
        else:
            self.test_waveform = False

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

    def _process_batch(self, batch, target_has_phys=False):
        additional_fields = None
        (c, f), target = batch
        if isinstance(f, list):
            additional_fields = f[1:]
            f = f[0]
        if self.use_fft:
            f = rfft(f)
        if self.write_script:
            self.write_model([c, f])
        if self.occlude_index:
            f[:, self.occlude_index] = 0
        predictions = self.model([c, f])
        if target_has_phys:
            loss, target_tensor, predictions, _ = \
                self._calc_segment_loss( c, predictions, target, target_index=self.evaluator.z_index)
        else:
            loss, target_tensor, predictions, _ = self._calc_segment_loss( c, predictions, target)
        return loss, predictions, target_tensor, c, f, additional_fields

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _, _ = self._process_batch(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, _, _, _ = self._process_batch(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.test_waveform:
            (c, f), target = batch
            f = cat((f, zeros((f.shape[0], 6), dtype=f.dtype)), dim=1)
            coo, f = create_coord_from_det(c, f)
            loss, predictions, target_tensor, c, f, _ = self._process_batch([(coo,f),target], self.test_has_phys)
            results_dict = {'test_loss': loss}
            if not self.evaluator.logger:
                self.evaluator.set_logger(self.logger)
            if self.test_has_phys:
                self.evaluator.add(predictions, target_tensor, c, f)
            else:
                self.evaluator.add(predictions, target_tensor, c, f, target_is_cal=self.target_is_cal)
            self.log_dict(results_dict, on_epoch=True, logger=True)
            return results_dict
        loss, predictions, target_tensor, c, f, additional_fields = self._process_batch(batch, self.test_has_phys)
        results_dict = {'test_loss': loss}
        if not self.evaluator.logger:
            self.evaluator.set_logger(self.logger)
        if self.test_has_phys:
            self.evaluator.add(predictions, target_tensor, c, f, additional_fields=additional_fields)
        else:
            self.evaluator.add(predictions, target_tensor, c, f, target_is_cal=self.target_is_cal, additional_fields=additional_fields)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict

