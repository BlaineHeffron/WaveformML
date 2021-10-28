from src.evaluation.AD1Evaluator import AD1Evaluator
import numpy as np
from numpy.fft import rfft
from torch import Tensor

from src.evaluation.MetricAggregator import MetricPairAggregator, MetricAggregator
from src.evaluation.PIDEvaluator import retrieve_class_names_PIDS, PID_MAP, PID_MAPPED_NAMES
from src.evaluation.RealDataEvaluator import convert_PID
from src.utils.SparseUtils import calc_calib_z_E
from src.utils.WaveformUtils import align_wfs

PULSE_ANALYSIS_SAMPLES = 5
NUM_Z_BINS = 10


class WaveformEvaluator(AD1Evaluator):
    def __init__(self, logger, calgroup=None, e_scale=None, **kwargs):
        super(WaveformEvaluator, self).__init__(logger, calgroup=calgroup, e_scale=e_scale, **kwargs)
        self.hascal = False
        self.sample_width = 4
        self.n_samples = 150
        self.t_center = np.arange(2, self.n_samples * self.sample_width - 1, self.sample_width)
        self.analyze_waveforms = "wf_analysis" in kwargs.keys()
        if "additional_field_names" in kwargs.keys():
            self.additional_field_names = kwargs["additional_field_names"]
            if "PID" in self.additional_field_names:
                self.PID_index = self.additional_field_names.index("PID")
                self.has_PID = True
        if self.analyze_waveforms:
            self.init_sample_metrics()

    def init_sample_metrics(self):
        metric_name = "z"
        metric_unit = "mm"
        scaling = 1.
        metric_names = ["sample {}".format(i) for i in range(PULSE_ANALYSIS_SAMPLES)]
        if hasattr(self, "has_PID") and self.has_PID:
            class_names, _ = retrieve_class_names_PIDS()
        else:
            class_names = ["any"]
        units = ["normalized ADC"] * PULSE_ANALYSIS_SAMPLES
        metric_params = [[0.0, 1.0, 100]] * PULSE_ANALYSIS_SAMPLES
        scales = [1.0] * PULSE_ANALYSIS_SAMPLES
        i = 0
        self.z_binned_metric_pairs = []
        for i in range(NUM_Z_BINS + 2):
            sample_metrics = []
            for name, unit, scale in zip(metric_names, units, scales):
                sample_metrics.append(MetricAggregator(name, *metric_params[i], class_names,
                                                       metric_name=metric_name, metric_unit=metric_unit,
                                                       scale_factor=scaling,
                                                       norm_factor=scale, parameter_unit=unit,
                                                       is_multiplicity=name == "multiplicity"))

                i += 1
            self.z_binned_metric_pairs.append(MetricPairAggregator(sample_metrics))

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

    def analyze_wf_z(self, wf, c, z, z_pred, additional_fields):
        has_PID = False
        if hasattr(self, "has_PID") and self.has_PID and additional_fields is not None:
            class_indices = additional_fields[self.PID_index]
            convert_PID(class_indices, PID_MAP)
            has_PID = True
        else:
            class_indices = np.zeros((c.shape[0]))
        wfs = self._align_wfs(wf)
        wfs = np.transpose(wfs, (2, 1, 0))
        inc = 1200 / NUM_Z_BINS
        results = np.abs(z - z_pred)
        for i in range(NUM_Z_BINS + 2):
            if has_PID:
                for j in range(len(PID_MAPPED_NAMES.keys())):
                    if i == 0:
                        inds = (class_indices == j) & (z <= -600)
                    elif i == 11:
                        inds = (class_indices == j) & (z >= 600)
                    elif i == 10:
                        inds = (class_indices == j) & (z > -600 + (i - 1) * inc) & (z < 600)
                    else:
                        inds = (class_indices == j) & (z > -600 + (i - 1) * inc) & (z <= -600 + i * inc)
                    self.z_binned_metric_pairs[i].add(results, wfs[:, 0, inds], PID_MAPPED_NAMES[j])
                    self.z_binned_metric_pairs[i].add(results, wfs[:, 1, inds], PID_MAPPED_NAMES[j])
            else:
                if i == 0:
                    inds = (z <= -600)
                elif i == 11:
                    inds = (z >= 600)
                elif i == 10:
                    inds = (z > -600 + (i - 1) * inc) & (z < 600)
                else:
                    inds = (z > -600 + (i - 1) * inc) & (z <= -600 + i * inc)
                self.z_binned_metric_pairs[i].add(results, wfs[:, 0, inds], "any")
                self.z_binned_metric_pairs[i].add(results, wfs[:, 1, inds], "any")

    def dump_wf_z(self):
        for i in range(NUM_Z_BINS + 2):
            if self.z_binned_metric_pairs[i] is not None:
                self.z_binned_metric_pairs[i].plot(self.logger)


    def _z_bin(self, z):
        inc = 1200 / NUM_Z_BINS
        if z <= -600:
            return 0
        elif z >= 600:
            return NUM_Z_BINS + 2
        for i in range(1, NUM_Z_BINS + 1):
            if -600 + i * inc > z:
                return i

    def fft_pulses(self, f: Tensor):
        wfs = self._align_wfs(f)
        return rfft(wfs)
