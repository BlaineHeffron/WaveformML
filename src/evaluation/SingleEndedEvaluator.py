from math import floor
import numpy as np

from src.evaluation.AD1Evaluator import AD1Evaluator


class SingleEndedEvaluator(AD1Evaluator):
    def __init__(self, logger, calgroup=None, e_scale=None):
        super().__init__(logger, calgroup=calgroup, e_scale=e_scale)
        SE_dead_pmts = [1, 0, 2, 4, 6, 7, 9, 10, 12, 13, 16, 19, 20, 21, 22, 24, 26, 27, 34, 36, 37, 43, 46, 48,
                        55,
                        54, 56, 58, 65, 68, 72, 80, 82, 85, 88, 93, 95, 97, 96, 105, 111, 112, 120, 122, 137, 138,
                        139, 141, 147, 158, 166, 173, 175, 188, 195, 215, 230, 243, 244, 245, 252, 255, 256, 261,
                        273, 279, 282]
        self.seg_status = np.zeros((self.nx, self.ny), dtype=np.float32)  # 0 for good, 0.5 for single ended, 1 for dead
        self.blind_detl = np.zeros((self.nx, self.ny), dtype=np.int8)  # 1 for blind , 0 for good
        self.blind_detr = np.zeros((self.nx, self.ny), dtype=np.int8)
        self.set_SE_segs(SE_dead_pmts)

    def set_SE_segs(self, SE_dead_pmts):
        for pmt in SE_dead_pmts:
            r = pmt % 2
            seg = int((pmt - r) / 2)
            x = seg % 14
            y = floor(seg / 14)
            self.seg_status[x, y] += 0.5
            if r == 0:
                self.blind_detl[x, y] = 1
            else:
                self.blind_detr[x, y] = 1
