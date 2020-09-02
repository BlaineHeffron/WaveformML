import numba as nb
from math import ceil, floor
from numba.typed import List


# TODO: implement this with pytorch + cython so it can stay on the gpu
@nb.jit(nopython=True)
def find_matches(pred, lab, out):
    for i in range(pred.shape[0]):
        if pred[i] == lab[i]:
            out[i] = 1
        else:
            out[i] = 0
    return out


@nb.jit(nopython=True)
def metric_accumulate_1d(metric, category, output, out_n, xrange, nbins):
    # expects output to be of size nbins+2, 1 for overflow and underflow
    xlen = xrange[1] - xrange[0]
    bin_width = xlen / nbins
    for i in range(metric.shape[0]):
        bin = 0
        find_bin = True
        if category[i] < xrange[0]:
            find_bin = False
        elif category[i] > xrange[1]:
            bin = nbins + 1
            find_bin = False
        if find_bin:
            for j in range(1, nbins + 1):
                if j * bin_width + xrange[0] > category[i]:
                    bin = j
                    break
        output[bin] += metric[i]
        out_n[bin] += 1

def get_typed_list(mylist):
    typed_list = List()
    [typed_list.append(x) for x in mylist]
    return typed_list

@nb.jit(nopython=True)
def metric_accumulate_2d(metric, category, output, out_n, xrange, yrange, nbinsx, nbinsy):
    # expects output to be of size nbins+2, 1 for overflow and underflow
    xlen = xrange[1] - xrange[0]
    ylen = yrange[1] - yrange[0]
    bin_widthx = xlen / nbinsx
    bin_widthy = ylen / nbinsy
    for i in range(metric.shape[0]):
        binx = 0
        biny = 0
        find_binx = True
        find_biny = True
        if category[i, 0] < xrange[0]:
            find_binx = False
        elif category[i, 0] > xrange[1]:
            binx = nbinsx + 1
            find_binx = False
        if category[i, 1] < yrange[0]:
            find_biny = False
        elif category[i, 1] > yrange[1]:
            biny = nbinsy + 1
            find_biny = False
        if find_binx:
            for j in range(1, nbinsx + 1):
                if j * bin_widthx + xrange[0] > category[i, 0]:
                    binx = j
                    break
        if find_biny:
            for j in range(1, nbinsy + 1):
                if j * bin_widthy + yrange[0] > category[i, 1]:
                    biny = j
                    break
        output[binx, biny] += metric[i]
        out_n[binx, biny] += 1


@nb.jit(nopython=True)
def average_pulse(coords, pulses, out_coords, out_pulses, multiplicity, psd):
    last_id = -1
    current_ind = -1
    n_current = 0
    psd_window_lo = -3
    psd_divider = 11
    psd_window_hi = 50
    for coord in coords:
        if coord[2] != last_id:
            if last_id > -1:
                out_coords[current_ind] /= n_current
                multiplicity[current_ind] = n_current
                psd[current_ind] = calc_psd(out_pulses[current_ind], calc_arrival(out_pulses[current_ind]),
                                            psd_window_lo, psd_window_hi, psd_divider)
            n_current = 0
            last_id = coord[2]
            current_ind += 1
        n_current += 1
        out_coords[current_ind] += coord[0:2]
        out_pulses[current_ind] += pulses[current_ind]
    out_coords[last_id] /= n_current
    multiplicity[current_ind] = n_current
    psd[current_ind] = calc_psd(out_pulses[current_ind], calc_arrival(out_pulses[current_ind]), psd_window_lo,
                                psd_window_hi, psd_divider)
    return out_coords, out_pulses, multiplicity, psd


@nb.jit(nopython=True)
def calc_arrival(fdat):
    peak = 0
    cur_ind = 0
    for d in fdat:
        if d > peak:
            peak = d
        cur_ind += 1
    cur_ind = 0
    for d in fdat:
        if d > 0.5 * peak:
            return cur_ind
        cur_ind += 1
    return 0


@nb.jit(nopython=True)
def calc_psd(fdat, arrival_samp, psd_window_lo, psd_window_hi, psd_divider):
    fast = integrate_lininterp_range(fdat, arrival_samp + psd_window_lo, arrival_samp + psd_divider)
    slow = integrate_lininterp_range(fdat, arrival_samp + psd_divider, arrival_samp + psd_window_hi)
    return slow / (slow + fast)


@nb.jit(nopython=True)
def integrate_lininterp_range(v, r0, r1):
    i0 = ceil(r0)
    d0 = i0 - r0
    i1 = floor(r1)
    d1 = r1 - i1
    if i0 <= i1:
        s = sum_range(v, i0, i1)
    else:
        s = 0
    if 0 <= i0 < v.size:
        s -= (1 - d0) * (1 - d0) / 2 * v[i0]
    if 1 <= i0 <= v.size:
        s += d0 * d0 / 2 * v[i0 - 1]
    if 0 <= i1 < v.size:
        s -= (1 - d1) * (1 - d1) / 2 * v[i1]
    if -1 <= i1 < v.size - 1:
        s += d1 * d1 / 2 * v[i1 + 1]
    return s


@nb.jit(nopython=True)
def sum_range(v, r0, r1):
    if r0 >= 0:
        r0 = r0
    else:
        r0 = 0
    if not r0 < v.size:
        return 0
    if r1 < v.size:
        r1 = r1
    else:
        r1 = v.size - 1
    if not (r0 <= r1):
        return 0
    sum = 0
    for i in range(r0, r1 + 1):
        sum += v[i]
    return sum
