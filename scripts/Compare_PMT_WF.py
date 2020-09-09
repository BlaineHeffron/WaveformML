import argparse, h5py
from pathlib import Path
from math import floor


class EventAdder:
    def __init__(self, coord, wf):
        self.evt_nums = []
        self.batch_nums = []
        self.current_event = Event()
        self.current_event.add_coord(coord, wf)
        self.batch_nums.append(coord[2])

    def add_coord(self, coord, wf):
        if self.current_event.batch_num != coord[2]:
            self.current_event.check_match()
            self.current_event = Event()
        self.current_event.add_coord(coord, wf)
        self.batch_nums.append(coord[2])

    def add_det(self, evt, det, h, a, b, rise, psd):
        self.current_event.add_det(det, h, a, b, rise, psd)
        self.evt_nums.append(evt)
        if len(self.evt_nums) != len(self.batch_nums):
            print("Error: difference in number of unique batch ids and unique evt ids\n")
            print("evt #: {0}, batch #: {1}".format(len(self.evt_nums), len(self.batch_nums)))

    def check_current(self):
        self.current_event.check_match()


class Event:
    def __init__(self):
        self.dets = []
        self.coords = []
        self.vals = []
        self.wfs = []
        self.batch_num = -1

    def add_coord(self, coord, wf):
        self.coords.append(coord)
        if self.batch_num == -1:
            self.batch_num = coord[2]
        elif self.batch_num != coord[2]:
            print("error - added coordinate from different batch: {0} vs {1}".format(self.batch_num, wf["coord"][2]))
        self.wfs.append(wf)

    def add_det(self, det, h, a, b, rise, psd):
        self.dets.append(det)
        self.vals.append((h, a, b, rise, psd))

    def check_match(self):
        for det in self.dets:
            r = det % 2
            seg = int((det - r) / 2)
            nx = seg % 14
            ny = int(floor(seg / 14))
            found_match = False
            for coord in self.coords:
                if coord[0] == nx:
                    if coord[1] == ny:
                        found_match = True
            return found_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mydir", help="directory to combined files or path to specific file")
    args = parser.parse_args()
    p = Path(args.mydir)
    all = p.glob("*.h5")
    wffiles = p.glob("*WaveformPairSim.h5")
    detfiles = p.glob("*PMTSim.h5")
    detfiles = [str(d) for d in detfiles]
    for wf in wffiles:
        det = str(wf.resolve()).replace("WaveformPair", "PMT")
        if not det in detfiles:
            print("didnt find match for " + str(wf.resolve()))
            for a in all:
                print(a.resolve())
                print()
            continue
        print("checking file {}".format(wf))
        ea = None
        with h5py.File(str(wf.resolve()), 'r') as h5f:
            with h5py.File(det, 'r') as h5det:
                coords = h5f["WaveformPairs"]["coord"]
                wfs = h5f["WaveformPairs"]["waveform"]
                evts = h5det["DetPulse"]["det"]
                dets = h5det["DetPulse"]["det"]
                area = h5det["DetPulse"]["a"]
                base = h5det["DetPulse"]["b"]
                height = h5det["DetPulse"]["h"]
                rise = h5det["DetPulse"]["rise"]
                psd = h5det["DetPulse"]["PSD"]
                for coord, wf, evt, det, a, b, h, r, p in zip(coords, wfs, evts, dets, area, base, height, rise, psd):
                    if ea is None:
                        ea = EventAdder(coord, wf)
                    else:
                        ea.add_coord(coord, wf)
                    ea.add_det(evt, det, a, b, h, r, p)
                ea.check_current()


if __name__ == "__main__":
    main()
