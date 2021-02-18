import torch

from src.evaluation.EnergyEvaluator import EnergyEvaluatorPhys, EnergyEvaluatorWF, EnergyEvaluatorBase
from src.evaluation.ZEvaluator import ZEvaluatorPhys, ZEvaluatorWF, ZEvaluatorBase
import numpy as np

from src.utils.SparseUtils import E_basic_prediction_dense


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
        EZEvaluatorBase.__init__(self, logger, e_scale=e_scale)
        self.EnergyEvaluator = EnergyEvaluatorPhys(logger, calgroup, e_scale)
        self.ZEvaluator = ZEvaluatorPhys(logger, e_scale=e_scale)
        if calgroup is not None:
            self.EnergyFromCalEval = EnergyEvaluatorPhys(logger, calgroup, e_scale, namespace="phys_z_pred")

    def add(self, predictions, target, c, f):
        self.EnergyEvaluator.add(predictions[:, 0, :, :].unsqueeze(1), target[:, 0, :, :].unsqueeze(1), c, f)
        self.ZEvaluator.add(predictions[:, 1, :, :].unsqueeze(1), target[:, 1, :, :].unsqueeze(1), c, f)
        if hasattr(self.EnergyEvaluator, "calibrator"):
            cal_E_pred = np.zeros(predictions[:, 0, :, :].shape)
            PE0 = f[:, self.EnergyEvaluator.PE0_index] * self.EnergyEvaluator.PE_scale
            PE1 = f[:, self.EnergyEvaluator.PE1_index] * self.EnergyEvaluator.PE_scale
            e = f[:, self.EnergyEvaluator.E_index] * self.EnergyEvaluator.E_scale
            dense_E = self.EnergyEvaluator.get_dense_matrix(
                torch.cat((e.unsqueeze(1), PE0.unsqueeze(1), PE1.unsqueeze(1)), dim=1), c, to_numpy=True)
            z_pred = (predictions[:, 1, :, :].detach().cpu().numpy() - 0.5) * self.EnergyEvaluator.z_scale
            E_basic_prediction_dense(dense_E, z_pred, self.EnergyEvaluator.nx, self.EnergyEvaluator.ny,
                                     self.ZEvaluator.seg_status,
                                     self.EnergyEvaluator.calibrator.light_pos_curves,
                                     self.EnergyEvaluator.calibrator.light_sum_curves, cal_E_pred)
            cal_E_pred = cal_E_pred / self.EnergyEvaluator.E_scale
            self.EnergyFromCalEval.add(np.expand_dims(cal_E_pred, 1), target[:, 0, :, :].unsqueeze(1), c, f, True)

    def dump(self):
        self.EnergyEvaluator.dump()
        self.ZEvaluator.dump()
        if hasattr(self, "EnergyFromCalEval"):
            self.EnergyFromCalEval.dump()

    def set_logger(self, l):
        self.logger = l
        self.EnergyEvaluator.logger = l
        self.ZEvaluator.logger = l
        if hasattr(self, "EnergyFromCalEval"):
            self.EnergyFromCalEval.logger = l


class EZEvaluatorWF(EZEvaluatorBase):
    def __init__(self, logger, calgroup=None, e_scale=None):
        super().__init__(logger, e_scale=e_scale)
        self.EnergyEvaluator = EnergyEvaluatorWF(logger, calgroup, e_scale)
        self.ZEvaluator = ZEvaluatorWF(logger, calgroup)
