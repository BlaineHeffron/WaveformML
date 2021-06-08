from src.engineering.LitBase import LitBase

class LitWaveform(LitBase):

    def __init__(self, config, trial=None):
        super(LitWaveform, self).__init__(config, trial)
        if config.net_config.net_class.endswith("RecurrentWaveformNet"):
            self.squeeze_index = 2
        else:
            self.squeeze_index = 1

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (c, f), target = batch
        predictions = self.model(f.unsqueeze(self.squeeze_index))
        loss = self.criterion.forward(predictions, target)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (c, f), target = batch
        predictions = self.model(f.unsqueeze(self.squeeze_index))
        loss = self.criterion.forward(predictions, target)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        (c, f), target = batch
        predictions = self.model(f.unsqueeze(self.squeeze_index))
        loss = self.criterion.forward(predictions, target)
        self.log('test_loss', loss, on_epoch=True, logger=True)
        if hasattr(self, "evaluator"):
            self.evaluator.add(batch, predictions)
        return loss

