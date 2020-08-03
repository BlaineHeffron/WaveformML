import numpy as np
import copy
from util import all_gather
import time


class PSDEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.psd_eval = {}
        self.event_ids = []

    def update(self, batch, predictions, labels):
        un = np.unique(batch[0][1][2], return_counts=True)
        import ipdb; ipdb.set_trace()
        event_ids = list(un[0])
        self.event_ids.extend(event_ids)
        return self.prepare(predictions, labels, un[1])

    def synchronize_between_processes(self):
        for event_type in self.event_types:
            self.eval_pulses[event_type] = np.concatenate(self.eval_pulses[event_type], 2)
            create_common_psd_eval(self.psd_eval[event_type], self.event_ids, self.eval_pulses[event_type])
