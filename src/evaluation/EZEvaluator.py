import torch

from src.evaluation.EnergyEvaluator import EnergyEvaluatorPhys, EnergyEvaluatorWF, EnergyEvaluatorBase
from src.evaluation.ZEvaluator import ZEvaluatorPhys, ZEvaluatorWF, ZEvaluatorBase
import spconv
import numpy as np

from src.utils.SparseUtils import E_basic_prediction


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
        self.ZEvaluator = ZEvaluatorPhys(logger)
        if calgroup is not None:
            self.EnergyFromCalEval = EnergyEvaluatorPhys(logger, calgroup, e_scale, namespace="phys_z_pred")

    def add(self, predictions, target, c, f):
        self.EnergyEvaluator.add(predictions[:, 0, :, :].unsqueeze(1), target[:, 0, :, :].unsqueeze(1), c, f)
        self.ZEvaluator.add(predictions[:, 1, :, :].unsqueeze(1), target[:, 1, :, :].unsqueeze(1), c, f)
        if hasattr(self.EnergyEvaluator, "calibrator"):
            cal_E_pred = np.zeros(f[:, self.EnergyEvaluator.E_index].shape)
            PE0 = f[:, self.EnergyEvaluator.PE0_index].detach().cpu().numpy() * self.EnergyEvaluator.PE_scale
            PE1 = f[:, self.EnergyEvaluator.PE1_index].detach().cpu().numpy() * self.EnergyEvaluator.PE_scale
            e = f[:, self.EnergyEvaluator.E_index].detach().cpu().numpy() * self.EnergyEvaluator.E_scale
            sparse_Z = spconv.SparseConvTensor.from_dense(predictions[:, 1, :, :].unsqueeze(1))
            permute_tensor = torch.tensor([1, 2, 0])
            coo = sparse_Z.indices[permute_tensor].detach().cpu().numpy()
            z_nn_preds = sparse_Z.features
            assert coo == c.detach().cpu().numpy()
            E_basic_prediction(sparse_Z.indices[permute_tensor].detach().cpu().numpy(), e, PE0, PE1, z_nn_preds,
                               self.ZEvaluator.seg_status, self.EnergyEvaluator.calibrator.light_pos_curves,
                               self.EnergyEvaluator.calibrator.light_sum_curves, cal_E_pred)
            cal_E_pred = cal_E_pred / self.EnergyEvaluator.E_scale
            cal_E_pred = self.EnergyEvaluator.get_dense_matrix(cal_E_pred, coo, to_numpy=False)
            self.EnergyFromCalEval.add(cal_E_pred, target[:, 0, :, :].unsqueeze(1), c, f)

    def dump(self):
        self.EnergyEvaluator.dump()
        self.ZEvaluator.dump()
        if hasattr(self, "EnergyFromCalEval"):
            self.EnergyFromCalEval.dump()


class EZEvaluatorWF(EZEvaluatorBase):
    def __init__(self, logger, calgroup=None, e_scale=None):
        super().__init__(logger, e_scale=e_scale)
        self.EnergyEvaluator = EnergyEvaluatorWF(logger, calgroup, e_scale)
        self.ZEvaluator = ZEvaluatorWF(logger, calgroup)
