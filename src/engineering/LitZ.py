import spconv
from src.models.SingleEndedZConv import SingleEndedZConv
from src.engineering.PSDDataModule import *
from torch import where, tensor, sum
from src.evaluation.ZEvaluator import ZEvaluatorWF, ZEvaluatorPhys


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
        self.SE_only = False
        if hasattr(self.net_config,"SELoss"):
            self.SE_only = self.net_config.SELoss
        if config.net_config.algorithm == "features":
            self.evaluator = ZEvaluatorPhys(self.logger)
        else:
            if hasattr(self.config.dataset_config, "calgroup"):
                self.evaluator = ZEvaluatorWF(self.logger, calgroup=self.config.dataset_config.calgroup)
            else:
                self.evaluator = ZEvaluatorWF(self.logger)
        if self.SE_only:
            self._format_SE_mask()

    def _format_SE_mask(self):
        self.SE_mask = tensor(self.evaluator.seg_status)
        for i in range(self.evaluator.nx):
            for j in range(self.evaluator.ny):
                if self.SE_mask[i,j] == 0.5:
                    self.SE_mask[i,j] = 1.0
                elif self.SE_mask[i, j] == 1.0:
                    self.SE_mask[i,j] = 0.
        self.SE_mask = self.SE_mask.unsqueeze(0)
        self.SE_mask = self.SE_mask.unsqueeze(0)
        self.SE_factor = (self.evaluator.nx*self.evaluator.ny) / sum(self.SE_mask)
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

    def _format_target_and_prediction(self, pred, coords, target, batch_size):
        target_tensor = spconv.SparseConvTensor(target.unsqueeze(1), coords[:, self.model.permute_tensor],
                                                self.model.spatial_size, batch_size)
        target_tensor = target_tensor.dense()
        # set output to 0 if there was no value for input
        return where(target_tensor == 0, target_tensor, pred), target_tensor

    def _process_batch(self, batch):
        (c, f), target = batch
        predictions = self.model([c, f])
        batch_size = c[-1, -1] + 1
        predictions, target_tensor = self._format_target_and_prediction(predictions, c, target, batch_size)
        if self.SE_only:
            loss = self.criterion.forward(self.SE_mask*predictions, self.SE_mask*target_tensor) * self.SE_factor
        else:
            loss = self.criterion.forward(predictions, target_tensor)
        loss *= (self.evaluator.nx*self.evaluator.ny*batch_size/c.shape[0])
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
        loss, predictions, target_tensor, c, f = self._process_batch(batch)
        results_dict = {'test_loss': loss}
        if not self.evaluator.logger:
           self.evaluator.logger = self.logger
        self.evaluator.add(predictions, target_tensor, c, f)
        self.log_dict(results_dict, on_epoch=True, logger=True)
        return results_dict
