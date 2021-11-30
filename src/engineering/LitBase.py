from os.path import join

from src.engineering.PSDDataModule import *
from torch import tensor, Tensor, ones, float32, sum
import spconv
import logging

from src.evaluation.SingleEndedEvaluator import SingleEndedEvaluator


class LitBase(pl.LightningModule):

    def __init__(self, config, trial=None, event_predictions=True):
        super(LitBase, self).__init__()
        if trial:
            self.trial = trial
        else:
            self.trial = None
        self.event_predictions = event_predictions
        self.nx = 14
        self.ny = 11
        self.pylog = logging.getLogger(__name__)
        logging.getLogger("lightning").setLevel(self.pylog.level)
        self.config = config
        if hasattr(config.system_config, "half_precision"):
            self.needs_float = not config.system_config.half_precision
        else:
            self.needs_float = True
        self.hparams.update(DictionaryUtility.to_dict(config))
        self.lr = config.optimize_config.lr
        self.modules = ModuleUtility(config.net_config.imports + config.dataset_config.imports +
                                     config.optimize_config.imports)
        if hasattr(config.net_config, "net_class"):
            self.model_class = self.modules.retrieve_class(config.net_config.net_class)
            self.model = self.model_class(config)
        else:
            self.model = None
        self.criterion_class = self.modules.retrieve_class(config.net_config.criterion_class)
        if self.event_predictions:
            criterion_named_params = {"reduction": 'mean'}
        else:
            criterion_named_params = {"reduction": 'sum'}
        self.criterion = self.criterion_class(*config.net_config.criterion_params, **criterion_named_params)
        self.write_script = False
        self.model_written = False
        if hasattr(config.dataset_config, "occlude_index"):
            self.occlude_index = config.dataset_config.occlude_index
        else:
            self.occlude_index = None
        self.SE_only = False
        if hasattr(self.config.net_config, "SELoss"):
            self.SE_only = self.config.net_config.SELoss
        if self.SE_only:
            self.evaluator = SingleEndedEvaluator(self.logger)
            self._format_SE_mask()

    def forward(self, x):
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

    def training_step(self, batch, batch_idx):
        data, target = batch
        predictions = self.model(data)
        loss = self.criterion.forward(predictions, target)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        predictions = self.model(data)
        loss = self.criterion.forward(predictions, target)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        if self.write_script:
            self.write_model(data)
        predictions = self.model(data)
        loss = self.criterion.forward(predictions, target)
        self.log('test_loss', loss, on_epoch=True, logger=True)
        if hasattr(self, "evaluator"):
            self.evaluator.add(batch, predictions)
        return loss

    def write_model(self, data):
        if not self.model_written:
            path = join(self.logger.experiment.log_dir, "model.pt")
            print("saving torchscript model to {}.".format(path))
            script_model = self.to_torchscript(path)
            print("saving model success")
            self.model_written = True

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
        self.register_buffer("SE_mask", SE_mask)
        print("Using single ended only loss.")

    def _calc_segment_loss(self, coo: Tensor, predictions: Tensor, target: Tensor, use_float=True, target_index=None, sparse_mask=None):
        """
        @param coo: tensor of coordinates, shape (batch size, 3)
        @param predictions: dense tensor of predictions, shape (# events, # outputs, x, y)
        @param target: tensor of target values still in sparse format, shape (batch size, 1)
        @param use_float: true if predicting continuous values with MSE or L1 loss, false if predicting classes
        @return  loss, sparse mask, target tensor, predictions
        Note, this assumes the criterion is using 'sum' as its aggregation method
        """
        batch_size = coo[-1, -1] + 1
        num_predictions = coo.shape[0]
        if target.shape[0] != num_predictions:
            raise ValueError("if using segment loss, target must have same number of elements in first dimension as coordinate tensor")
        if sparse_mask is None:
            sparse_mask = spconv.SparseConvTensor(ones((num_predictions, predictions.shape[1]), dtype=float32, device=self.device),
                                              coo[:, self.model.permute_tensor],
                                              self.model.spatial_size, batch_size).dense()
        if len(target.shape) == 1:
            target_tensor = spconv.SparseConvTensor(target.unsqueeze(1), coo[:, self.model.permute_tensor],
                                                self.model.spatial_size, batch_size).dense()
        else:
            target_tensor = spconv.SparseConvTensor(target, coo[:, self.model.permute_tensor],
                                                    self.model.spatial_size, batch_size).dense()
        predictions = sparse_mask * predictions
        if self.SE_only:
            if target_index is None:
                if use_float:
                    loss = self.criterion.forward(self.SE_mask * predictions, self.SE_mask * target_tensor)
                else:
                    loss = self.criterion.forward(self.SE_mask * predictions, self.SE_mask * target_tensor.squeeze(1))
            else:
                if use_float:
                    loss = self.criterion.forward(self.SE_mask * predictions,
                                                  self.SE_mask * target_tensor[:, target_index, :, :].unsqueeze(1))
                else:
                    loss = self.criterion.forward(self.SE_mask * predictions,
                                                  self.SE_mask * target_tensor[:, target_index, :, :])
        else:
            if target_index is None:
                if use_float:
                    loss = self.criterion.forward(predictions, target_tensor)
                else:
                    loss = self.criterion.forward(predictions, target_tensor.squeeze(1))
            else:
                if use_float:
                    loss = self.criterion.forward(predictions, target_tensor[:, target_index, :, :].unsqueeze(1))
                else:
                    loss = self.criterion.forward(predictions, target_tensor[:, target_index, :, :])
        if self.SE_only:
            num_predictions = sum(self.SE_mask * sparse_mask)
        return loss / num_predictions, target_tensor, predictions, sparse_mask
