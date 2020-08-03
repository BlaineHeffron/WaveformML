import numpy as np
import copy
from util import all_gather
import time


class PSDEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.psd_eval = {}
        self.event_ids = []

    def update(self, predictions):
        event_ids = list(np.unique(list(predictions.keys())))
        self.event_ids.extend(event_ids)
        return self.prepare(predictions)

    def synchronize_between_processes(self):
        for event_type in self.event_types:
            self.eval_pulses[event_type] = np.concatenate(self.eval_pulses[event_type], 2)
            create_common_psd_eval(self.psd_eval[event_type], self.event_ids, self.eval_pulses[event_type])

    def accumulate(self):
        for psd_eval in self.psd_eval.values():
            psd_eval.evaluate()
            psd_eval.accumulate()

    def prepare(self, predictions):
        psd_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            psd_results.extend(
                [
                    {
                        "event_id": original_id,
                        "category_id": labels,
                        "score": scores
                    }
                ]
            )
        return psd_results

def merge(event_ids, eval_pulses):
    all_event_ids = all_gather(event_ids)
    all_eval_pulses = all_gather(eval_pulses)

    merged_event_ids = []
    for p in all_event_ids:
        merged_event_ids.extend(p)

    merged_eval_pulses = []
    for p in all_eval_pulses:
        merged_eval_pulses.append(p)

    merged_event_ids = np.array(merged_event_ids)
    merged_eval_pulses = np.concatenate(merged_eval_pulses, 2)

    # keep only unique (and in sorted order) events
    merged_event_ids, idx = np.unique(merged_event_ids, return_index=True)
    merged_eval_pulses = merged_eval_pulses[..., idx]

    return merged_event_ids, merged_eval_pulses


def create_common_psd_eval(psd_eval, event_ids, eval_pulses):
    event_ids, eval_pulses = merge(event_ids, eval_pulses)
    event_ids = list(event_ids)
    eval_pulses = list(eval_pulses.flatten())

    psd_eval.evalpulses = eval_pulses
    psd_eval.params.pulseIds = event_ids
    psd_eval._paramsEval = copy.deepcopy(psd_eval.params)

