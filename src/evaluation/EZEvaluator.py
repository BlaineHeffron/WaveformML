from src.evaluation.EnergyEvaluator import EnergyEvaluatorPhys, EnergyEvaluatorWF, EnergyEvaluatorBase
from src.evaluation.ZEvaluator import ZEvaluatorPhys, ZEvaluatorWF, ZEvaluatorBase


class EZEvaluatorBase:
    def __init__(self, logger):
        self.logger = logger
        self.EnergyEvaluator = EnergyEvaluatorBase(logger)
        self.ZEvaluator = ZEvaluatorBase(logger)

    def add(self, predictions, target, c, f):
        self.EnergyEvaluator.add(predictions[:, 0, :, :].unsqueeze(1), target[:, 0, :, :].unsqueeze(1), c, f)
        self.ZEvaluator.add(predictions[:, 1, :, :].unsqueeze(1), target[:, 1, :, :].unsqueeze(1), c, f)

    def dump(self):
        self.EnergyEvaluator.dump()
        self.ZEvaluator.dump()


class EZEvaluatorPhys(EZEvaluatorBase):
    def __init__(self, logger):
        super().__init__(logger)
        self.EnergyEvaluator = EnergyEvaluatorPhys(logger)
        self.ZEvaluator = ZEvaluatorPhys(logger)


class EZEvaluatorWF(EZEvaluatorBase):
    def __init__(self, logger, calgroup=None):
        super().__init__(logger)
        self.EnergyEvaluator = EnergyEvaluatorWF(logger, calgroup)
        self.ZEvaluator = ZEvaluatorWF(logger, calgroup)
