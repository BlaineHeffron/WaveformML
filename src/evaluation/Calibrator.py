
import logging
from src.utils.SQLUtils import CalibrationDB
import numpy as np
from math import floor


def get_coords_from_chan(chan):
    r = chan % 2
    seg = int((chan - r) / 2)
    nx = seg % 14
    ny = floor(seg / 14)
    return nx, ny, r


def log_ratio_points(s0, s1, zmin, zmax, npts, vx, vy):
    for i in range(npts):
        vy[i] = zmin + i * (zmax - zmin) / float(npts - 1)
    vx_num = s1.eval(vy)
    vx_den = s0.eval(vy)
    vx_new = np.log(np.divide(vx_num, vx_den))
    for i in range(npts):
        vx[i] = vx_new[i]


def eval_sum(g0, g1, s0, s1, zmin, zmax, npts, vx, vy):
    for i in range(npts):
        vx[i] = zmin + i * (zmax - zmin) / float(npts - 1)
    vy_new = s0 * g0.eval(vx) + s1 * g1.eval(vx)
    for i in range(npts):
        vy[i] = vy_new[i]


class Calibrator:
    def __init__(self, calibdb: CalibrationDB):
        """
        @type calibdb: object
        """
        self.log = logging.getLogger(__name__)
        self.calibdb = calibdb
        self.gains, self.eres, self.rel_times, self.seg_times = calibdb.get_seg_cal_values()
        atten_curves, lsum_curves, time_curves, lin_curves, self.psd_curves, t_interp_curves, \
        self.e_ncapt = calibdb.get_curves()
        self.sampletime = np.zeros((14, 11, 2), dtype=np.float32)
        self.light_pos_curves = np.zeros((14, 11, 51, 2), dtype=np.float32)
        self.time_pos_curves = np.zeros((14, 11, 50, 2), dtype=np.float32)
        self.light_sum_curves = np.zeros((14, 11, 50, 2), dtype=np.float32)
        self.t_interp_curves = np.zeros((14, 11, 2, 50, 2), dtype=np.float32)
        self.calc_light_pos_curve(atten_curves)
        self.calc_time_pos_curve(time_curves)
        self.calc_light_sum_curve(lsum_curves, atten_curves)
        self.calc_t_interp_curve(t_interp_curves)
        for chan, curve in t_interp_curves.items():
            if curve:
                nx, ny, r = get_coords_from_chan(chan)
                self.sampletime[nx, ny, r] = round(max(curve.xs))

    def calc_t_interp_curve(self, t_interp_curves):
        for chan in range(14 * 11 * 2):
            if chan in t_interp_curves.keys():
                if t_interp_curves[chan]:
                    t_interp_curves[chan].sort()
                    nx, ny, r = get_coords_from_chan(chan)
                    xs = np.linspace(t_interp_curves[chan].xs[0],t_interp_curves[chan].xs[-1],50)
                    self.t_interp_curves[nx, ny, r, :, 0] = xs
                    self.t_interp_curves[nx, ny, r, :, 1] = t_interp_curves.eval(xs)

    def calc_light_pos_curve(self, atten_curves):
        for seg in range(14 * 11):
            l = seg * 2
            r = seg * 2 + 1
            if l in atten_curves.keys() and r in atten_curves.keys():
                vx = np.zeros((51,), dtype=np.float32)
                vy = np.zeros((51,), dtype=np.float32)
                curvel, curver = atten_curves[l], atten_curves[r]
                if not curvel or not curver:
                    continue
                curvel.sort()
                curver.sort()
                nx, ny, _ = get_coords_from_chan(l)
                zmin = max([curvel.xs[0], curver.xs[0]])
                zmax = min([curvel.xs[-1], curver.xs[-1]])
                self.log.debug("light ratio curve pmt l seg {0} is {1}".format(seg,curvel))
                self.log.debug("light ratio curve pmt r curve r is {}".format(curver))
                log_ratio_points(curvel, curver, zmin, zmax, 51, vx, vy)
                self.log.debug("light pos curve vx is is {}".format(vx))
                self.log.debug("light pos curve vy is {}".format(vy))
                self.light_pos_curves[nx, ny, :, 0] = vx
                self.light_pos_curves[nx, ny, :, 1] = vy

    def calc_time_pos_curve(self, time_curves):
        npts = 50
        for seg in range(14 * 11):
            l = seg * 2
            r = seg * 2 + 1
            if l in time_curves.keys() and r in time_curves.keys():
                vy = np.zeros((npts,), dtype=np.float32)
                curvel, curver = time_curves[l], time_curves[r]
                curvel.sort()
                curver.sort()
                nx, ny, _ = get_coords_from_chan(l)
                zmin = max([curvel.xs[0], curver.xs[0]])
                zmax = min([curvel.xs[-1], curver.xs[-1]])
                assert (zmin < zmax)
                for i in range(npts):
                    vy[i] = zmax + i * (zmin - zmax) / float(npts - 1)
                vx = curver.eval(vy) - curvel.eval(vy)
                self.time_pos_curves[nx, ny, :, 0] = vx
                self.time_pos_curves[nx, ny, :, 1] = vy

    def calc_light_sum_curve(self, lsum_curves, atten_curves):
        npts = 50
        for seg in range(14 * 11):
            l = seg * 2
            r = seg * 2 + 1
            if l not in lsum_curves.keys():
                if l in atten_curves.keys():
                    lsum_curves[l] = atten_curves[l]
            if r not in lsum_curves.keys():
                if r in atten_curves.keys():
                    lsum_curves[r] = atten_curves[r]
            if l in lsum_curves.keys() and r in lsum_curves.keys():
                nx, ny, _ = get_coords_from_chan(l)
                vx = np.zeros((npts,), dtype=np.float32)
                vy = np.zeros((npts,), dtype=np.float32)
                eval_sum(lsum_curves[l], lsum_curves[r], self.eres[nx, ny, 0], self.eres[nx, ny, 1], -650, 650, npts,
                         vx, vy)
                self.light_sum_curves[nx, ny, :, 0] = vx
                self.light_sum_curves[nx, ny, :, 1] = vy

