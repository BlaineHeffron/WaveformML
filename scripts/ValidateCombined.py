import h5py
import argparse
from pathlib import Path
import json
from numpy import asarray
import numba as nb


def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


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
        feats = ds["waveform"]
        inds = _npwhere((coords[:, 2] >= file_info[1][0]) & (
                coords[:, 2] <= file_info[1][1]))
    return coords[inds], feats[inds]


def _npwhere(cond):
    return asarray(cond).nonzero()


def check_file(j, merged_coords, merged_feats, y, n, f):
    ind = 0
    event_ind = 0
    skip_current = False
    current_batch_coord = 0
    batch_to_skip = []
    ylen = y.shape[0]
    for fdat in j[str(n)]:
        coords, feats = _read_chunk(fdat)
        print("checking against file {}".format(fdat))
        for coord, feat in zip(coords,feats):
            #print("sanity check: y =  {0}, n = {1}, y==n : {2}".format(y[event_ind], n,
            #    y[event_ind]==n))
            prev_batch_coord = current_batch_coord
            current_batch_coord = merged_coords[ind,2]
            #print("prev batch: {0}, current batch:"
            #      " {1}".format(prev_batch_coord,current_batch_coord))
            if skip_current:
                if current_batch_coord in batch_to_skip:
                    continue
                else:
                    skip_current = False
            prev_event_ind = event_ind
            if prev_batch_coord != current_batch_coord:
                #print("prev batch: {0}, current batch:"
                #      " {1}".format(prev_batch_coord,current_batch_coord))
                event_ind += 1
            while ylen > event_ind and y[event_ind] != n:
                #print("ylen: {0}, event_ind: {1}, y: {2}".format(ylen, event_ind, y[event_ind]))
                event_ind += 1
            if event_ind >= ylen:
                break
            traverse_next = event_ind - prev_event_ind
            #print("traverse next: {}".format(traverse_next))
            if traverse_next > 0:
                skip_current = True
                batch_to_skip = [coord[2] + i for i in range(traverse_next)]
                #print("batch to skip: {0}".format(batch_to_skip))
                continue
            #print("checking for row {0} event {1}".format(ind,event_ind))
            #print("orig coords: {0} merged coords: {1}".format(coord[0:2],merged_coords[ind,0:2]))
            if not (arrays_equal(coord[0:2], merged_coords[ind, 0:2])):
                raise ValueError("File {0} contained incorrect coords at index {1}: \n"
                                 "Value: {2}\nExpected: {3}".format(str(f.resolve()), ind, merged_coords[ind, 0:2],
                                                                    coord[0:2]))
            if not (arrays_equal(feat, merged_feats[ind, :])):
                raise ValueError("File {0} contained incorrect waveform at index {1}: \n"
                                 "Value: {2}\nExpected: {3}".format(str(f.resolve()), ind, merged_feats[ind,:],
                                                                    feat))
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
            merged_feats = h5f["WaveformPairs"]["waveform"]
            y = h5f["WaveformPairs"]["labels"]
            j = json_load(str(f.resolve())[0:-3] + ".json")
            for n in j.keys():
                check_file(j, merged_coords, merged_feats, y, int(n), f)


if __name__ == "__main__":
    main()
