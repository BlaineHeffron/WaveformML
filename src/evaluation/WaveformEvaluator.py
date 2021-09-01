import os

from src.evaluation.AD1Evaluator import AD1Evaluator
import numpy as np
from numpy.fft import rfft
from torch import Tensor, arange, argmax, tensor

from src.utils.SparseUtils import calc_calib_z_E
from src.utils.WaveformUtils import align_wfs

PULSE_ANALYSIS_SAMPLES = 20


class WaveformEvaluator(AD1Evaluator):
    def __init__(self, logger, calgroup=None, e_scale=None, **kwargs):
        super(WaveformEvaluator, self).__init__(logger, calgroup=calgroup, e_scale=e_scale, **kwargs)
        self.hascal = False
        self.sample_width = 4
        self.n_samples = 150
        self.t_center = np.arange(2, self.n_samples * self.sample_width - 1, self.sample_width)

    def z_E_from_cal(self, c, f, shape):
        Z = np.zeros(shape, dtype=np.float32)
        E = np.zeros(shape, dtype=np.float32)
        calc_calib_z_E(c, f, Z, E, self.sample_width, self.calibrator.t_interp_curves, self.calibrator.sampletime,
                       self.calibrator.rel_times, self.gain_factor, self.calibrator.eres,
                       self.calibrator.time_pos_curves, self.calibrator.light_pos_curves,
                       self.calibrator.light_sum_curves, self.z_scale, self.n_samples)
        return Z, E

    def _align_wfs(self, f):
        f = f.reshape((f.shape[0], 2, f.shape[1] / 2))
        wfs = np.zeros((f.shape[0], 2, PULSE_ANALYSIS_SAMPLES))
        f = f.detach().cpu().numpy()
        align_wfs(f, wfs)
        return wfs

    def fft_pulses(self, f: Tensor):
        wfs = self._align_wfs(f)
        return rfft(wfs)
