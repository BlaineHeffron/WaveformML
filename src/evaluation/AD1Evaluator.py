import os
from typing import Dict

import spconv
import torch
import numpy as np

from src.datasets.HDF5Dataset import MAX_RANGE
from src.evaluation.Calibrator import Calibrator
from src.utils.SQLUtils import CalibrationDB
from src.utils.SQLiteUtils import get_gains

E_NORMALIZATION_FACTOR = 12.
Z_NORMALIZATION_FACTOR = 1200.
CELL_LENGTH = 1176.


class AD1Evaluator:
    """
    for physcoord definition:
    vs[0] = p.E / 12.;
    vs[1] = (p.dt / 30.) + 0.5;
    vs[2] = p.PE[0] / 5000.;
    vs[3] = p.PE[1] / 5000.;
    vs[4] = (p.z / 1200.0) + 0.5;
    vs[5] = p.PSD;
    vs[6] = ((Float_t)(p.t - toffset)) / 30.;
    """
    def __init__(self, calgroup=None, e_scale=None):
        self.nx = 14
        self.ny = 11
        self.spatial_size = np.array([self.nx, self.ny])
        self.permute_tensor = torch.LongTensor([2, 0, 1])  # needed because spconv requires batch index first
        self.z_scale = Z_NORMALIZATION_FACTOR
        self.E_scale = E_NORMALIZATION_FACTOR
        if e_scale:
            self.E_adjust = self.E_scale / e_scale
            self.E_scale = e_scale
        else:
            self.E_adjust = 1.0

        self.dt_scale = 30.
        self.toffset_scale = 30.
        self.PE_scale = 5000. / self.E_adjust
        self.dp_scale = CELL_LENGTH
        self.E_index = 0
        self.dt_index = 1
        self.PE0_index = 2
        self.PE1_index = 3
        self.z_index = 4
        self.PSD_index = 5
        self.toffset_index = 6
        self.dp_index = 7
        self.phys_names = ["Energy", "dt", "PE0", "PE1", "z", "PSD", "t offset", "distance to PMT"]
        self.phys_units = ["MeV", "ns", "", "", "mm", "", "ns", "mm"]
        self.default_bins = [[0.0, self.E_scale, 40], [-self.dt_scale/2., self.dt_scale/2., 40], [0.0, self.PE_scale, 40],
                             [0.0, self.PE_scale, 40], [-self.z_scale/2., self.z_scale/2., 40], [0.0, 1.0, 40], [0.0, self.toffset_scale, 40], [0.0, CELL_LENGTH, 40]]
        if calgroup is not None:
            self.hascal = True
            if "PROSPECT_CALDB" not in os.environ.keys():
                raise ValueError(
                    "Error: could not find PROSPECT_CALDB environment variable. Please set PROSPECT_CALDB to be the "
                    "path of the sqlite3 calibration database.")
            gains = get_gains(os.environ["PROSPECT_CALDB"], calgroup)
            self.gain_factor = np.divide(np.full((self.nx, self.ny, 2), MAX_RANGE), gains)
            self.calibrator = Calibrator(CalibrationDB(os.environ["PROSPECT_CALDB"], calgroup))

    def override_default_bins(self, bin_overrides: Dict):
        for key in bin_overrides.keys():
            try:
                self.default_bins[int(key)] = bin_overrides[key]
            except ValueError:
                raise IOError("Keys for 'evaluation_config.bin_overrides' dictionary must be integers")


    def get_dense_matrix(self, data: torch.tensor, c: torch.tensor, to_numpy=True):
        batch_size = c[-1, -1] + 1
        if data.dim() == 1:
            data = spconv.SparseConvTensor(data.unsqueeze(1), c[:, self.permute_tensor],
                                           self.spatial_size, batch_size)
        else:
            data = spconv.SparseConvTensor(data, c[:, self.permute_tensor],
                                           self.spatial_size, batch_size)
        data = data.dense()
        if to_numpy:
            data = data.detach().cpu().numpy()
        return data

    def scale_factor(self, index):
        if index == self.E_index:
            return self.E_scale
        elif index == self.dt_index:
            return self.dt_scale
        elif index == self.PE0_index:
            return self.PE_scale
        elif index == self.PE1_index:
            return self.PE_scale
        elif index == self.z_index:
            return self.z_scale
        elif index == self.PSD_index:
            return 1.0
        elif index == self.toffset_index:
            return self.toffset_scale
        elif index == self.dp_index:
            return self.dp_scale

