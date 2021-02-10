import spconv
import torch
import numpy as np


class AD1Evaluator:
    def __init__(self):
        self.nx = 14
        self.ny = 11
        self.spatial_size = np.array([self.nx, self.ny])
        self.permute_tensor = torch.LongTensor([2, 0, 1])  # needed because spconv requires batch index first
        self.z_scale = 1200.
        self.E_scale = 300.


class PhysCoordEvaluator(AD1Evaluator):
    """
    for physcoord definition:
    vs[0] = p.E / 300.;
    vs[1] = (p.dt / 200.) + 0.5;
    vs[2] = p.PE[0] / 125000.;
    vs[3] = p.PE[1] / 125000.;
    vs[4] = (p.z / 1200.0) + 0.5;
    vs[5] = p.PSD;
    vs[6] = ((Float_t)(p.t - toffset)) / 600.;
    """
    def __init__(self):
        super(PhysCoordEvaluator, self).__init__()
        self.dt_scale = 200.
        self.toffset_scale = 600.
        self.PE_scale = 125000.
        self.E_index = 0
        self.dt_index = 1
        self.PE0_index = 2
        self.PE1_index = 3
        self.z_index = 4
        self.PSD_index = 5
        self.toffset_index = 6

    def get_dense_matrix(self, data: torch.tensor, c: torch.tensor):
        batch_size = c[-1, -1] + 1
        data = spconv.SparseConvTensor(data.unsqueeze(1), c[:, self.permute_tensor],
                                       self.spatial_size, batch_size)
        data = data.dense()
        data = data.detach().cpu().numpy()
        return data
