import h5py
import argparse
from pathlib import Path
import json
from numpy import asarray
from src.utils.util import json_load
import numba as nb


@nb.jit(nopython=True)
def arrays_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


def _read_chunk(file_info):
    with h5py.File(file_info[0], 'r', ) as h5_file:
        ds = h5_file["WaveformPairs"]
        coords = ds["coord"]
        inds = _npwhere((coords[:, 2] >= file_info[1][0]) & (
                coords[:, 2] <= file_info[1][1]))
    return coords[inds]


def _npwhere(cond):
    return asarray(cond).nonzero()


def check_file(j, merged_coords, y, n, f):
    ind = 0
    event_ind = 0
    skip_current = False
    current_batch_coord = 0
    batch_to_skip = []
    ylen = y.shape[0]
    for fdat in j[str(n)]:
        coords = _read_chunk(fdat)
        for coord in coords:
            prev_batch_coord = current_batch_coord
            current_batch_coord = coord[2]
            if skip_current:
                if current_batch_coord in batch_to_skip:
                    continue
                else:
                    skip_current = False
            prev_event_ind = event_ind
            if prev_batch_coord != current_batch_coord:
                event_ind += 1
            while (ylen > event_ind and y[event_ind] != n):
                event_ind += 1
            if event_ind >= ylen:
                break
            traverse_next = event_ind - prev_event_ind
            if traverse_next > 0:
                skip_current = True
                batch_to_skip = [coord[2] + i for i in range(traverse_next)]
                continue
            if not (arrays_equal(coord[0:2], merged_coords[ind, 0:2])):
                raise ValueError("File {0} contained incorrect value at index {1}: \n"
                                 "Value: {2}\nExpected: {3}".format(str(f.resolve()), ind, merged_coords[ind, 0:2],
                                                                    coord[0:2]))
            ind += 1
        if event_ind >= ylen:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mydir", help="directory to combined files or path to specific file")
    args = parser.parse_args()
    p = Path(args.mydir)
    if p.is_file():
        files = [p]
    else:
        files = p.glob("*.h5")
    for f in files:
        print("checking file {}".format(f))
        with h5py.File(str(f.resolve()), 'r') as h5f:
            merged_coords = h5f["WaveformPairs"]["coord"]
            y = h5f["WaveformPairs"]["labels"]
            j = json_load(str(f.resolve())[0:-3] + ".json")
            for n in j.keys():
                check_file(j, merged_coords, y, int(n), f)


if __name__ == "__main__":
    main()
