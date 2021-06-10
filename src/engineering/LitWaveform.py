from src.engineering.LitBase import LitBase
from src.evaluation.TensorEvaluator import TensorEvaluator
from src.utils.util import DictionaryUtility


class LitWaveform(LitBase):

    def __init__(self, config, trial=None):
        super(LitWaveform, self).__init__(config, trial)
        if config.net_config.net_class.endswith("RecurrentWaveformNet"):
            self.squeeze_index = 2
        else:
            self.squeeze_index = 1
        if hasattr(self.config.dataset_config, "test_dataset_params"):
            if self.config.dataset_config.test_dataset_params.label_name == "phys" and not hasattr(
                    self.config.dataset_config.test_dataset_params, "label_index"):
                self.test_has_phys = True
        if hasattr(self.config.dataset_config, "calgroup"):
            calgroup = self.config.dataset_config.calgroup
        else:
            calgroup = None
        self.target_index = self.config.dataset_config.label_index
        self.evaluator = TensorEvaluator(self.logger, calgroup=calgroup,
                                         target_has_phys=self.test_has_phys, target_index=target_index)
        self.loss_no_reduce = self.modules.retrieve_class(self.config.optimize_config.optimizer_class)(self.model.parameters(),
                                                                                     lr=(self.lr or self.learning_rate),
                                                                                     **DictionaryUtility.to_dict(
                                                                                         self.config.optimize_config.optimizer_params),
                                                                                                       reduction='none')
    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (c, f), target = batch
        predictions = self.model(f.unsqueeze(self.squeeze_index)).squeeze(1)
        loss = self.criterion.forward(predictions, target)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (c, f), target = batch
        predictions = self.model(f.unsqueeze(self.squeeze_index)).squeeze(1)
        loss = self.criterion.forward(predictions, target)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        (c, f), target = batch
        predictions = self.model(f.unsqueeze(self.squeeze_index)).squeeze(1)
        if self.test_has_phys:
            loss = self.criterion.forward(predictions, target[:,self.target_index])
        else:
            loss = self.criterion.forward(predictions, target)
        self.log('test_loss', loss, on_epoch=True, logger=True)
        if hasattr(self, "evaluator"):
            results = self.loss_no_reduce(predictions, target)
            self.evaluator.add(target, results)
        return loss

