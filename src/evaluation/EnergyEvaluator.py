from src.evaluation.AD1Evaluator import PhysCoordEvaluator
from src.evaluation.SingleEndedEvaluator import SingleEndedEvaluator
from src.evaluation.WaveformEvaluator import WaveformEvaluator
from src.utils.StatsUtils import StatsAggregator
from src.utils.SparseUtils import E_deviation, E_deviation_with_z


class EnergyEvaluatorBase(StatsAggregator, SingleEndedEvaluator):

    def __init__(self, logger):
        super(EnergyEvaluatorBase, self).__init__(logger)
        SingleEndedEvaluator.__init__(self)
        self.hascal = False
        self.E_bounds = [0., 12.]
        self.mult_bounds = [0.5, 10.5]
        self.n_mult = 10
        self.n_E = 20
        self.n_z = 20
        self.z_bounds = [-600., 600.]
        self.E_mult_names = ["E_mult_single", "E_mult_single_cal", "E_mult_dual", "E_mult_dual_cal"]
        self.E_mult_titles = ["Single Ended", "Single Ended", "Double Ended", "Double Ended"]
        self.E_z_names = ["E_z_single", "E_z_single_cal", "E_z_dual", "E_z_dual_cal"]
        self.seg_mult_names = ["seg_mult_Emae", "seg_mult_Emae_cal"]
        self.initialize()

    def initialize(self):
        self.register_duplicates(self.E_mult_names,
                                 [self.n_E, self.n_mult], [self.E_bounds[0], self.mult_bounds[0]],
                                 [self.E_bounds[1], self.mult_bounds[1]], 2, ["Energy", "Multiplicity"], ["MeV", ""],
                                 "Energy MAE", "MeV", underflow=(1, 0), scale=self.E_scale)
        self.register_duplicates(self.E_z_names, [self.n_E, self.n_z],
                                 [self.E_bounds[0], self.z_bounds[0]],
                                 [self.E_bounds[1], self.z_bounds[1]], 2, ["Energy", "Z Position"], ["MeV", "mm"],
                                 "Energy MAE", "MeV", scale=self.E_scale)
        self.register_duplicates(self.seg_mult_names, [self.nx, self.ny, self.n_mult],
                                 [0.5, 0.5, 0.5],
                                 [self.nx + 0.5, self.ny + 0.5, self.n_mult + 0.5], 3,
                                 ["x segment", "y segment", "Multiplicity"], [""] * 3, "Energy MAE", "MeV",
                                 underflow=False, overflow=(0, 0, 1), scale=self.E_scale)

    def calc_deviation_with_z(self, pred, targ, E, Z):
        E_deviation_with_z(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_Emae"][0],
                           self.results["seg_mult_Emae"][1], self.results["E_mult_dual"][0],
                           self.results["E_mult_dual"][1], self.results["E_mult_single"][0],
                           self.results["E_mult_single"][1], self.seg_status, self.nx, self.ny,
                           self.n_mult, self.n_E, self.E_bounds[0], self.E_bounds[1], self.E_scale,
                           self.z_scale, Z, self.results["E_z_dual"][0],
                           self.results["E_z_dual"][1], self.results["E_z_single"][0],
                           self.results["E_z_single"][1])
        E_deviation_with_z(E, targ[:, 0, :, :], self.results["seg_mult_Emae_cal"][0],
                           self.results["seg_mult_Emae_cal"][1], self.results["E_mult_dual_cal"][0],
                           self.results["E_mult_dual_cal"][1], self.results["E_mult_single_cal"][0],
                           self.results["E_mult_single_cal"][1], self.seg_status, self.nx, self.ny,
                           self.n_mult, self.n_E, self.E_bounds[0], self.E_bounds[1], self.E_scale,
                           self.z_scale, Z, self.results["E_z_dual_cal"][0],
                           self.results["E_z_dual_cal"][1], self.results["E_z_single_cal"][0],
                           self.results["E_z_single_cal"][1])

    def dump(self):
        for name, title in zip(self.E_mult_names, self.E_mult_titles):
            self.log_total(name, "evaluation/{}".format(name), title)
            self.log_metric(name, "evaluation/{}".format(name), title)
        for name, title in zip(self.E_z_names, self.E_mult_titles):
            self.log_total(name, "evaluation/{}".format(name), title)
            self.log_metric(name, "evaluation/{}".format(name), title)
        for name in self.seg_mult_names:
            self.log_segment_metric(name, "evaluation/{}".format(name))

    def add(self, predictions, target, c, f):
        pass


class EnergyEvaluatorWF(EnergyEvaluatorBase, WaveformEvaluator):
    def __init__(self, logger, calgroup=None):
        super().__init__(logger)
        super().__init__(calgroup=calgroup)

    def add(self, predictions, target, c, f):
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        if self.hascal:
            c, f = c.detach().cpu().numpy(), f.detach().cpu().numpy()
            Z, E = self.z_E_from_cal(c, f, (pred.shape[0], pred.shape[2], pred.shape[3]))
            self.calc_deviation_with_z(pred, targ, E, Z)

        else:
            E_deviation(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_Emae"][0],
                        self.results["seg_mult_Emae"][1], self.results["E_mult_dual"][0],
                        self.results["E_mult_dual"][1], self.results["E_mult_single"][0],
                        self.results["E_mult_single"][1], self.seg_status, self.nx, self.ny,
                        self.n_mult, self.n_E, self.E_bounds[0], self.E_bounds[1], self.E_scale)


class EnergyEvaluatorPhys(EnergyEvaluatorBase, PhysCoordEvaluator):
    def __init__(self, logger):
        super(EnergyEvaluatorPhys, self).__init__(logger)
        PhysCoordEvaluator.__init__(self)

    def add(self, predictions, target, c, f):
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        Z = self.get_dense_matrix(f[:, self.z_index], c)
        E = self.get_dense_matrix(f[:, self.E_index], c)
        self.calc_deviation_with_z(pred, targ, E, Z)
