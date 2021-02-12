from src.evaluation.EnergyEvaluator import EnergyEvaluatorPhys, EnergyEvaluatorWF, EnergyEvaluatorBase
from src.evaluation.ZEvaluator import ZEvaluatorPhys, ZEvaluatorWF, ZEvaluatorBase


class EZEvaluatorBase:
    def __init__(self, logger, e_scale=None):
        self.logger = logger
        self.EnergyEvaluator = EnergyEvaluatorBase(logger, e_scale=e_scale)
        self.ZEvaluator = ZEvaluatorBase(logger)

    def add(self, predictions, target, c, f):
        self.EnergyEvaluator.add(predictions[:, 0, :, :].unsqueeze(1), target[:, 0, :, :].unsqueeze(1), c, f)
        self.ZEvaluator.add(predictions[:, 1, :, :].unsqueeze(1), target[:, 1, :, :].unsqueeze(1), c, f)

    def dump(self):
        self.EnergyEvaluator.dump()
        self.ZEvaluator.dump()

    def set_logger(self, l):
        self.logger = l
        self.EnergyEvaluator.logger = l
        self.ZEvaluator.logger = l

class EZEvaluatorPhys(EZEvaluatorBase):
    def __init__(self, logger, calgroup=None, e_scale=None):
        super().__init__(logger, e_scale)
        self.EnergyEvaluator = EnergyEvaluatorPhys(logger, calgroup, e_scale)
        self.ZEvaluator = ZEvaluatorPhys(logger)


class EZEvaluatorWF(EZEvaluatorBase):
    def __init__(self, logger, calgroup=None, e_scale=None):
        super().__init__(logger, e_scale)
        self.EnergyEvaluator = EnergyEvaluatorWF(logger, calgroup, e_scale)
        self.ZEvaluator = ZEvaluatorWF(logger, calgroup)
