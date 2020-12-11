import sqlite3
from math import floor
from numpy import zeros, float32
from scipy.interpolate import splrep, splev


class SQLiteBase:

    def __init__(self, path):
        self.path = path
        self.__db_connection = sqlite3.connect(path)
        self.cur = self.__db_connection.cursor()

    def close(self):
        self.__db_connection.close()

    def execute(self, new_data):
        self.cur.execute(new_data)

    def fetchone(self, sql):
        self.execute(sql)
        result = self.cur.fetchone()
        return result

    def fetchall(self, sql):
        self.execute(sql)
        result = self.cur.fetchall()
        return result

    def create_table(self, name, collist):
        txt = "CREATE TABLE IF NOT EXISTS {0}(".format(name)
        i = 0
        for col in collist:
            if i == 0:
                txt += col
            else:
                txt += ',' + col
        txt += ')'
        self.cur.execute(txt)

    def commit(self):
        self.__db_connection.commit()

    def __del__(self):
        self.__db_connection.close()

    def __enter__(self):
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        self.cur.close()
        if isinstance(exc_value, Exception):
            self.__db_connection.rollback()
        else:
            self.__db_connection.commit()
        self.__db_connection.close()


class OptunaDB(SQLiteBase):

    def __init__(self, path):
        super(OptunaDB, self).__init__(path)

    def get_best_trial(self):
        """returns trial number - 1 so you can go into the correct directory
        (directory names start with 0, trial indices in sqlite start with 1)"""

        results = self.fetchall(
            "select trial_id, study_id, value from trials WHERE value IS NOT NULL order by value asc limit 10")
        print("Top 10 trials: \n")
        for r in results:
            print("{0}: {1}".format(r[0], r[2]))
        return results[0][0] - 1


class CalibrationDB(SQLiteBase):

    def __init__(self, path, calgroup):
        super(CalibrationDB, self).__init__(path)
        self.calgroup = calgroup
        self.calgroup_id = None
        results = self.fetchall("select object_id from named_object where name = '{}'".format(self.calgroup))
        for r in results:
            self.calgroup_id = r[0]
            break

    def get_gains(self):
        return self.get_seg_cal_values()[0]

    def get_seg_cal_values(self):
        if not self.calgroup_id:
            return
        results = self.fetchall(
            "SELECT seg, lgain_0, lgain_1, eres_0, eres_1, rel_time, seg_time from segment_response where calgroup_id = '{}'".format(
                self.calgroup_id))
        gains = zeros((14, 11, 2), dtype=float32)
        eres = zeros((14, 11, 2), dtype=float32)
        rel_times = zeros((14, 11), dtype=float32)
        seg_times = zeros((14, 11), dtype=float32)
        for r in results:
            seg = int(r[0])
            nx = seg % 14
            ny = floor(seg / 14)
            gains[nx, ny, 0] = abs(r[1])
            gains[nx, ny, 1] = abs(r[2])
            eres[nx, ny, 0] = r[3]
            eres[nx, ny, 1] = r[4]
            rel_times[nx, ny] = r[5]
            seg_times[nx, ny] = r[6]
        return gains, eres, rel_times, seg_times

    def get_curves(self):
        if not self.calgroup_id:
            return
        atten_curves = {}
        lsum_curves = {}
        time_curves = {}
        lin_curves = {}
        psd_curves = {}
        t_interp_curves = {}
        e_ncapt = zeros((14, 11, 2), dtype=float32)
        pmt_response_id = None
        for r in self.fetchall(
                "SELECT pmt_response_id FROM calibration_group WHERE object_id = {}".format(self.calgroup_id)):
            pmt_response_id = r[0]
        if pmt_response_id:
            for r in self.fetchall(
                    "SELECT chan, atten_curve_id, lsum_curve_id, time_curve_id, linearity_curve_id, psd_curve_id, t_interp_curve_id, E_ncapt FROM pmt_response WHERE object_id = {}".format(
                            pmt_response_id)):
                if r[0]:
                    chan = int(r[0])
                    atten_curves[chan] = self.get_cal_curve(r[1])
                    lsum_curves[chan] = self.get_cal_curve(r[2])
                    time_curves[chan] = self.get_cal_curve(r[3])
                    lin_curves[chan] = self.get_cal_curve(r[4])
                    psd_curves[chan] = self.get_cal_curve(r[5])
                    t_interp_curves[chan] = self.get_cal_curve(r[6])
                    remainder = chan % 2
                    seg = int((chan - remainder) / 2)
                    e_ncapt[seg % 14, floor(seg / 14), remainder] = r[7]
        return atten_curves, lsum_curves, time_curves, lin_curves, psd_curves, t_interp_curves, e_ncapt

    def get_cal_curve(self, obj_id):
        if not obj_id:
            return
        curve = CalCurve()
        for r in self.fetchall("SELECT x,y,dx,dy FROM graph_points WHERE object_id = {}".format(obj_id)):
            curve.add_point(r[0], r[1], r[2], r[3])
        return curve


class CalCurve:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.xerr = []
        self.yerr = []
        self.spline = None

    def add_point(self, x, y, dx, dy):
        self.xs.append(x)
        self.ys.append(y)
        self.xerr.append(dx)
        self.yerr.append(dy)

    def sort(self):
        self.xs, self.ys, self.xerr, self.yerr = zip(*sorted(zip(self.xs, self.ys, self.xerr, self.yerr)))

    def get_spline(self):
        if 0 in self.yerr:
            self.spline = splrep(self.xs, self.ys)
        else:
            self.spline = splrep(self.xs, self.ys, w=[1. / y for y in self.yerr])

    def eval(self, x):
        if not self.spline:
            self.get_spline()
        return splev(x, self.spline)
