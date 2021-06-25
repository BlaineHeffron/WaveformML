import spconv
from src.models.SingleEndedZConv import SingleEndedZConv
from src.engineering.PSDDataModule import *
from torch import where, tensor, sum, cat, zeros, int32, floor_divide
from torch.fft import rfft
from src.evaluation.ZEvaluator import ZEvaluatorWF, ZEvaluatorPhys, ZEvaluatorRealWFNorm


def create_coord_from_det(c, f):
    coord = zeros((f.shape[0], 3), dtype=int32)
    coord[:, 0] = c % 14
    floor_divide(c, 14, out=coord[:, 1])
    features = zeros((f.shape[0],f.shape[1]*2), dtype=f.dtype)
    n_samp = f.shape[1]
    for i in range(coord.shape[0]):
        coord[i, 2] = i
        if c[i] % 2 == 0:
            features[i,0:n_samp] = f[i]
        else:
            features[i,n_samp:] = f[i]

    return coord, features


class LitZ(pl.LightningModule):

    def __init__(self, config, trial=None):
        super(LitZ, self).__init__()
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
        self.model = SingleEndedZConv(self.config)
        self.criterion_class = self.modules.retrieve_class(config.net_config.criterion_class)
        self.criterion = self.criterion_class(*config.net_config.criterion_params)
        self.test_has_phys = False
        if hasattr(self.config.dataset_config, "test_dataset_params"):
            if self.config.dataset_config.test_dataset_params.label_name == "phys" and not hasattr(
                    self.config.dataset_config.test_dataset_params, "label_index"):
                self.test_has_phys = True
        if hasattr(self.config.net_config, "UseFFT"):
            print("Using FFT")
            self.use_fft = True
        else:
            self.use_fft = False
        self.SE_only = False
        if hasattr(self.config.net_config, "SELoss"):
            self.SE_only = self.config.net_config.SELoss
        if self.test_has_phys:
            if hasattr(self.config.dataset_config, "calgroup"):
                self.evaluator = ZEvaluatorRealWFNorm(self.logger, calgroup=self.config.dataset_config.calgroup)
            else:
                self.evaluator = ZEvaluatorRealWFNorm(self.logger)
        elif config.net_config.algorithm == "features":
            self.evaluator = ZEvaluatorPhys(self.logger)
        else:
            if hasattr(self.config.dataset_config, "calgroup"):
                self.evaluator = ZEvaluatorWF(self.logger, calgroup=self.config.dataset_config.calgroup)
            else:
                self.evaluator = ZEvaluatorWF(self.logger)
        if self.SE_only:
            self._format_SE_mask()
        if self.config.dataset_config.dataset_class == "PulseDatasetRealWFPair":
            self.target_is_cal = True
        else:
            self.target_is_cal = False
        if self.config.dataset_config.dataset_class == "PulseDatasetWaveformNorm":
            self.target_is_cal = True
            self.test_waveform = True
        else:
            self.test_waveform = False

    def _format_SE_mask(self):
        SE_mask = tensor(self.evaluator.seg_status)
        for i in range(self.evaluator.nx):
            for j in range(self.evaluator.ny):
                if SE_mask[i, j] == 0.5:
                    SE_mask[i, j] = 1.0
                elif SE_mask[i, j] == 1.0:
                    SE_mask[i, j] = 0.
        SE_mask = SE_mask.unsqueeze(0)
        SE_mask = SE_mask.unsqueeze(0)
        self.SE_factor = (self.evaluator.nx * self.evaluator.ny) / sum(SE_mask)
        self.register_buffer("SE_mask", SE_mask)
        print("Using single ended only loss.")

    def forward(self, x, *args, **kwargs):
        return self.model(x)

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

    def _format_target_and_prediction(self, pred, coords, target, batch_size, target_has_phys=False):
        if target_has_phys:
            target_tensor = spconv.SparseConvTensor(target, coords[:, self.model.permute_tensor],
                                                    self.model.spatial_size, batch_size)
        else:
            target_tensor = spconv.SparseConvTensor(target.unsqueeze(1), coords[:, self.model.permute_tensor],
                                                    self.model.spatial_size, batch_size)
        target_tensor = target_tensor.dense()
        # set output to 0 if there was no value for input
        if target_has_phys:
            return where(target_tensor[:, self.evaluator.z_index, :, :] == 0,
                         target_tensor[:, self.evaluator.z_index, :, :], pred[:, 0, :, :]).unsqueeze(1), target_tensor
        else:
            return where(target_tensor == 0, target_tensor, pred), target_tensor

    def _process_batch(self, batch, target_has_phys=False):
        (c, f), target = batch
        if self.use_fft:
            f = rfft(f)
        predictions = self.model([c, f])
        batch_size = c[-1, -1] + 1
        predictions, target_tensor = self._format_target_and_prediction(predictions, c, target, batch_size,
                                                                        target_has_phys)
        if target_has_phys:
            if self.SE_only:
                loss = self.criterion.forward(self.SE_mask * predictions[:, 0, :, :],
                                              self.SE_mask * target_tensor[:, self.evaluator.z_index, :,
                                                             :]) * self.SE_factor
            else:
                loss = self.criterion.forward(predictions[:, 0, :, :], target_tensor[:, self.evaluator.z_index, :, :])
        else:
            if self.SE_only:
                loss = self.criterion.forward(self.SE_mask * predictions, self.SE_mask * target_tensor) * self.SE_factor
            else:
                loss = self.criterion.forward(predictions, target_tensor)
        loss *= (self.evaluator.nx * self.evaluator.ny * batch_size / c.shape[0])
        return loss, predictions, target_tensor, c, f

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self._process_batch(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, _, _ = self._process_batch(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.test_waveform:
            (c, f), target = batch
            f = cat((f, zeros((f.shape[0], 6), dtype=f.dtype)))
            coo, f = create_coord_from_det(c, f)
            loss, predictions, target_tensor, c, f = self._process_batch([(coo,f),target], self.test_has_phys)
            results_dict = {'test_loss': loss}
            if not self.evaluator.logger:
                self.evaluator.set_logger(self.logger)
            if self.test_has_phys:
                self.evaluator.add(predictions, target_tensor, c)
            else:
                self.evaluator.add(predictions, target_tensor, c, f, target_is_cal=self.target_is_cal)
            self.log_dict(results_dict, on_epoch=True, logger=True)
            return results_dict
        loss, predictions, target_tensor, c, f = self._process_batch(batch, self.test_has_phys)
        results_dict = {'test_loss': loss}
        if not self.evaluator.logger:
            self.evaluator.set_logger(self.logger)
        if self.test_has_phys:
            self.evaluator.add(predictions, target_tensor, c)
        else:
            self.evaluator.add(predictions, target_tensor, c, f, target_is_cal=self.target_is_cal)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict

