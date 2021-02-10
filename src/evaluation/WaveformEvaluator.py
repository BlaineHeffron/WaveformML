import os

from src.datasets.HDF5Dataset import MAX_RANGE
from src.evaluation.AD1Evaluator import AD1Evaluator
from src.evaluation.Calibrator import Calibrator
from src.utils.SQLUtils import CalibrationDB
from src.utils.SQLiteUtils import get_gains
import numpy as np

from src.utils.SparseUtils import calc_calib_z_E


class WaveformEvaluator(AD1Evaluator):
    def __init__(self, calgroup=None):
        super(WaveformEvaluator, self).__init__()
        self.hascal = False
        self.sample_width = 4
        self.n_samples = 150

        if calgroup is not None:
            self.hascal = True
            if "PROSPECT_CALDB" not in os.environ.keys():
                raise ValueError(
                    "Error: could not find PROSPECT_CALDB environment variable. Please set PROSPECT_CALDB to be the "
                    "path of the sqlite3 calibration database.")
            gains = get_gains(os.environ["PROSPECT_CALDB"], calgroup)
            self.gain_factor = np.divide(np.full((self.nx, self.ny, 2), MAX_RANGE), gains)
            self.t_center = np.arange(2, self.n_samples * self.sample_width - 1, self.sample_width)
            self.calibrator = Calibrator(CalibrationDB(os.environ["PROSPECT_CALDB"], calgroup))

    def z_E_from_cal(self, c, f, shape):
        Z = np.zeros(shape, dtype=np.float32)
        E = np.zeros(shape, dtype=np.float32)
        calc_calib_z_E(c, f, Z, E, self.sample_width, self.calibrator.t_interp_curves, self.calibrator.sampletime,
                       self.calibrator.rel_times, self.gain_factor, self.calibrator.eres,
                       self.calibrator.time_pos_curves, self.calibrator.light_pos_curves,
                       self.calibrator.light_sum_curves, self.z_scale, self.n_samples)
        return Z, E
