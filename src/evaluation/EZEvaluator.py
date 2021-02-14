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
            PE0 = f[:, self.EnergyEvaluator.PE0_index] * self.EnergyEvaluator.PE_scale
            PE1 = f[:, self.EnergyEvaluator.PE1_index] * self.EnergyEvaluator.PE_scale
            e = f[:, self.EnergyEvaluator.E_index] * self.EnergyEvaluator.E_scale
            dense_E = self.EnergyEvaluator.get_dense_matrix(torch.cat((e.unsqueeze(1), PE0.unsqueeze(1), PE1.unsqueeze(1)), dim=1), c, to_numpy=False)
            sparse_ZE = spconv.SparseConvTensor.from_dense(torch.cat((predictions[:, 1, :, :].unsqueeze(1), dense_E), dim=1))
            permute_tensor = torch.tensor([1, 2, 0])
            coo = sparse_ZE.indices[permute_tensor].detach().cpu().numpy()
            z_nn_preds = sparse_ZE.features[:, 0].detach().cpu().numpy()
            e = sparse_ZE.features[:, 1].detach().cpu().numpy()
            PE0 = sparse_ZE.features[:, 2].detach().cpu().numpy()
            PE1 = sparse_ZE.features[:, 3].detach().cpu().numpy()
            E_basic_prediction(coo, e, PE0, PE1, z_nn_preds, self.ZEvaluator.seg_status,
                               self.EnergyEvaluator.calibrator.light_pos_curves,
                               self.EnergyEvaluator.calibrator.light_sum_curves, cal_E_pred)
            cal_E_pred = cal_E_pred / self.EnergyEvaluator.E_scale
            cal_E_pred = self.EnergyEvaluator.get_dense_matrix(torch.tensor(cal_E_pred), torch.tensor(coo), to_numpy=False)
            self.EnergyFromCalEval.add(cal_E_pred[:, 0, :, :], target[:, 0, :, :].unsqueeze(1), c, f)

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
