import numba as nb
from math import ceil, floor, sqrt, log, exp
from numba.typed import List
from numpy import zeros, int32, float32
from src.utils.NumbaFunctions import merge_sort_two, merge_sort_main_numba
from src.datasets.HDF5Dataset import MAX_RANGE


# TODO: implement this with pytorch + cython so it can stay on the gpu

@nb.jit(nopython=True)
def moment(data, n, weights=None):
    ave, adev, sdev, svar, skew, curt = 0, 0, 0, 0, 0, 0
    if n <= 1:
        return svar, skew, curt
    s = 0.
    weightsum = 0.
    for j in range(n):
        if weights is not None:
            if weights[j] > 0:
                s += data[j] * weights[j]
                weightsum += weights[j]
        else:
            s += data[j]
    if weightsum > 0.0:
        ave = s / weightsum
    else:
        ave = s / n
    for j in range(n):
        if data[j]:
            s = data[j] - ave
            if weightsum > 0.0 and weights is not None:
                adev += abs(s) * weights[j]
                p = s * s
                svar += p * weights[j]
                p *= s
                skew += p * weights[j]
                curt += (p * s) * weights[j]
            else:
                adev += abs(s)
                p = s * s
                svar += p
                p *= s
                skew += p
                curt += p * s

    if weightsum > 0.0 and weights is not None:
        adev /= weightsum
        if weightsum > 1.:
            svar /= (weightsum - 1)
        else:
            svar = 0
        sdev = sqrt(svar)
        if svar:
            skew /= (weightsum * svar * sdev)
            curt = (curt / (weightsum * svar * svar)) - 3.0
    else:
        adev /= n
        if n > 1:
            svar /= (n - 1)
        else:
            svar = 0
        sdev = sqrt(svar)
        if svar:
            skew /= (n * svar * sdev)
            curt = (curt / (n * svar * svar)) - 3.0
    return svar, skew, curt


@nb.jit(nopython=True)
def safe_divide(a, b):
    for i in range(a.shape[0]):
        if b[i] == 0:
            a[i] = 0
        else:
            a[i] = a[i] / b[i]
    return a


def safe_divide_2d(a, b):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if b[i, j] == 0:
                a[i, j] = 0
            else:
                a[i, j] = a[i, j] / b[i, j]
    return a


@nb.jit(nopython=True)
def find_matches(pred, lab, out):
    for i in range(pred.shape[0]):
        if pred[i] == lab[i]:
            out[i] = 1
        else:
            out[i] = 0
    return out


@nb.jit(nopython=True)
def vec_sum(a):
    out = 0
    for val in a:
        out += val
    return out


@nb.jit(nopython=True)
def confusion_accumulate_1d(prediction, label, metric, output, xrange, nbins):
    xlen = xrange[1] - xrange[0]
    bin_width = xlen / nbins
    for i in range(metric.shape[0]):
        bin_index = 0
        find_bin = True
        if metric[i] < xrange[0]:
            find_bin = False
        elif metric[i] > xrange[1]:
            bin_index = nbins + 1
            find_bin = False
        if find_bin:
            for j in range(1, nbins + 1):
                if j * bin_width + xrange[0] > metric[i]:
                    bin_index = j - 1
                    break
            output[bin_index, label[i], prediction[i]] += 1


@nb.jit(nopython=True)
def get_bin_index(val, low, high, bin_width, nbins):
    bin_index = 0
    find_bin = True
    if val < low:
        find_bin = False
    elif val >= high:
        bin_index = nbins + 1
        find_bin = False
    if find_bin:
        for j in range(1, nbins + 1):
            if j * bin_width + low > val:
                bin_index = j
                break
    return bin_index


@nb.jit(nopython=True)
def metric_accumulate_1d(results, metric, output, out_n, xrange, nbins):
    # expects output to be of size nbins+2, 1 for overflow and underflow
    xlen = xrange[1] - xrange[0]
    bin_width = xlen / nbins
    for i in range(results.shape[0]):
        bin_index = 0
        find_bin = True
        if metric[i] < xrange[0]:
            find_bin = False
        elif metric[i] >= xrange[1]:
            bin_index = nbins + 1
            find_bin = False
        if find_bin:
            for j in range(1, nbins + 1):
                if j * bin_width + xrange[0] > metric[i]:
                    bin_index = j
                    break
        output[bin_index] += results[i]
        out_n[bin_index] += 1


def get_typed_list(mylist):
    typed_list = List()
    [typed_list.append(x) for x in mylist]
    return typed_list


@nb.jit(nopython=True)
def metric_accumulate_2d(results, metric, output, out_n, xrange, yrange, nbinsx, nbinsy):
    # expects output to be of size nbins+2, 1 for overflow and underflow
    xlen = xrange[1] - xrange[0]
    ylen = yrange[1] - yrange[0]
    bin_widthx = xlen / nbinsx
    bin_widthy = ylen / nbinsy
    for i in range(results.shape[0]):
        binx = 0
        biny = 0
        find_binx = True
        find_biny = True
        if metric[i, 0] < xrange[0]:
            find_binx = False
        elif metric[i, 0] >= xrange[1]:
            binx = nbinsx + 1
            find_binx = False
        if metric[i, 1] < yrange[0]:
            find_biny = False
        elif metric[i, 1] >= yrange[1]:
            biny = nbinsy + 1
            find_biny = False
        if find_binx:
            for j in range(1, nbinsx + 1):
                if j * bin_widthx + xrange[0] > metric[i, 0]:
                    binx = j
                    break
        if find_biny:
            for j in range(1, nbinsy + 1):
                if j * bin_widthy + yrange[0] > metric[i, 1]:
                    biny = j
                    break
        output[binx, biny] += results[i]
        out_n[binx, biny] += 1


@nb.jit(nopython=True)
def normalize_coords(out_coord, tot_l_current, tot_r_current, psdl, psdr, dt):
    if tot_l_current > 0 or tot_r_current > 0:
        dt /= (tot_l_current + tot_r_current)
        out_coord /= (tot_l_current + tot_r_current)
    if tot_l_current > 0:
        psdl /= tot_l_current
    if tot_r_current > 0:
        psdr /= tot_r_current
    return out_coord, psdl, psdr, dt


@nb.jit(nopython=True)
def calc_spread(coords, pulses, nsamp, mult, x, y, dt, E):
    dx = 0
    dy = 0
    ddt = 0
    dE = 0
    tot = 0
    if mult < 2:
        return dx, dy, ddt, dE
    for i in range(mult):
        totl = 0
        totr = 0
        timel = 0
        timer = 0
        for j in range(nsamp * 2):
            if j < nsamp:
                timel += pulses[i, j] * (j + 0.5)
                totl += pulses[i, j]
            else:
                timer += pulses[i, j] * (j - nsamp + 0.5)
                totr += pulses[i, j]
        tot += totl + totr
        if totl > 0 and totr > 0:
            ddt += abs((timer / totr - timel / totl) - dt) * (totl + totr)
            dE += abs(E - (totl + totr))
        elif totl > 0:
            ddt += abs(-1.0 * timel / totl - dt) * totl
            dE += abs(E - totl)
        elif totr > 0:
            ddt += abs(timer / totr - dt) * totr
            dE += abs(E - totr)
        dx += abs(coords[i, 0] - x) * (totl + totr)
        dy += abs(coords[i, 1] - y) * (totl + totr)
    if tot > 0:
        return dx / tot, dy / tot, ddt / tot, dE / mult
    else:
        return 0, 0, 0, 0


@nb.jit(nopython=True)
def calc_time(pulse, nsamp):
    """ returns energy weighted time in units of samples"""
    t = 0.0
    mysum = 0.0
    for i in range(nsamp):
        t += pulse[i] * (i + 0.5)
        mysum += pulse[i]
    if mysum != 0.0:
        return t / mysum
    else:
        return 0


@nb.jit(nopython=True)
def find_max(v):
    max = 0
    i = 0
    max_loc = 0
    for val in v:
        if val > max:
            max = val
            max_loc = i
        i += 1
    return max_loc


@nb.jit(nopython=True)
def average_pulse(coords, pulses, gains, times, out_coords, out_pulses, out_stats, multiplicity, psdl, psdr):
    """units for dx, dy are in cell widths, dt is in sample length,
    ddt is spread in dt, dt is time difference between left and right PMTs"""
    last_id = -1
    current_ind = -1
    n_current = 0
    tot_l_current = 0
    tot_r_current = 0
    dt_current = 0
    E_current = 0
    psd_window_lo = -3
    psd_divider = 11
    psd_window_hi = 50
    n_samples = pulses.shape[1] / 2
    pulse_ind = 0
    for coord in coords:
        if coord[2] != last_id:
            if last_id > -1:
                E_current /= n_current
                out_coords[current_ind], psdl[current_ind], psdr[current_ind], dt_current = normalize_coords(
                    out_coords[current_ind], tot_l_current, tot_r_current, psdl[current_ind], psdr[current_ind],
                    dt_current)
                out_stats[0, current_ind], out_stats[1, current_ind], out_stats[2, current_ind], out_stats[
                    3, current_ind] = calc_spread(
                    coords[pulse_ind - n_current:pulse_ind], pulses[pulse_ind - n_current:pulse_ind], n_samples,
                    n_current, out_coords[current_ind, 0], out_coords[current_ind, 1], dt_current, E_current)
                pulse = out_pulses[current_ind, 0:n_samples] + out_pulses[current_ind, n_samples:]
                out_stats[4, current_ind], _, _ = moment(times, n_samples, weights=pulse)
                out_stats[5, current_ind], _, _ = moment(pulse, n_samples)
                multiplicity[current_ind] = n_current
            n_current = 0
            tot_l_current = 0
            tot_r_current = 0
            dt_current = 0
            E_current = 0
            last_id = coord[2]
            current_ind += 1
        n_current += 1
        pulseleft = pulses[pulse_ind, 0:n_samples] * gains[coord[0], coord[1], 0]
        pulseright = pulses[pulse_ind, n_samples:2 * n_samples] * gains[coord[0], coord[1], 1]
        # TODO find peaks
        maxlocr = find_max(pulseright)
        # baselineright = find_baseline(pulseright, maxlocr, -30, -5)
        baselineright = 0
        ra_right = get_residual(baselineright)
        maxlocl = find_max(pulseleft)
        pulses[pulse_ind, 0:n_samples] = pulseleft
        # baselineleft = find_baseline(pulseleft, maxlocl, -30, -5)
        baselineleft = 0
        ra_left = get_residual(baselineleft)
        pulses[pulse_ind, n_samples:2 * n_samples] = pulseright
        tot_l = vec_sum(pulseleft)
        tot_r = vec_sum(pulseright)
        tot_l_current += tot_l
        tot_r_current += tot_r
        psdl[current_ind] += calc_psd(pulseleft, calc_arrival(pulseleft),
                                      psd_window_lo, psd_window_hi, psd_divider, ra_left) * tot_l
        psdr[current_ind] += calc_psd(pulseright, calc_arrival(pulseright),
                                      psd_window_lo, psd_window_hi, psd_divider, ra_right) * tot_r
        dt_current += (calc_time(pulseright, n_samples) - calc_time(pulseleft, n_samples)) * (tot_l + tot_r)
        E_current += tot_l + tot_r
        out_coords[current_ind] += coord[0:2] * (tot_l + tot_r)
        out_pulses[current_ind] += pulses[pulse_ind]
        pulse_ind += 1

    E_current /= n_current
    out_coords[current_ind], psdl[current_ind], psdr[current_ind], dt_current = normalize_coords(
        out_coords[current_ind], tot_l_current, tot_r_current, psdl[current_ind], psdr[current_ind], dt_current)
    out_stats[0, current_ind], out_stats[1, current_ind], out_stats[2, current_ind], out_stats[
        3, current_ind] = calc_spread(
        coords[pulse_ind - n_current:pulse_ind], pulses[pulse_ind - n_current:pulse_ind], n_samples,
        n_current, out_coords[current_ind, 0], out_coords[current_ind, 1], dt_current, E_current)
    pulse = out_pulses[current_ind, 0:n_samples] + out_pulses[current_ind, n_samples:]
    out_stats[4, current_ind], _, _ = moment(times, n_samples, weights=pulse)
    out_stats[5, current_ind], _, _ = moment(pulse, n_samples)
    multiplicity[current_ind] = n_current


@nb.jit(nopython=True)
def weighted_average_quantities(coords, full_quantities, out_quantities, out_coords, out_mult, n):
    """
    full_quantities is  features vector with first dimension the feature, second dimension the entries
    assumed energy is at index 0 and psd at 1, multiplicity (vector of 1s) at index n-1
    out_quantities is np.array of shape (number features, batch size)
    out_coords is array of zeros of shape (batch size, 2)
    n is number of features in full_quantities/out_quantities list
    """
    last_id = -1
    current_ind = -1
    n_current = 0
    ene_current = 0.0
    quant_ind = 0
    for coord in coords:
        if coord[2] != last_id:
            if last_id > -1:
                if ene_current > 0:
                    out_coords[current_ind] /= ene_current
                    for j in range(1, n):
                        out_quantities[j, current_ind] /= ene_current
                    out_mult[current_ind] = n_current
                    out_quantities[0, current_ind] = ene_current
            n_current = 0
            ene_current = 0.0
            last_id = coord[2]
            current_ind += 1
        n_current += 1
        ene_current += full_quantities[0, quant_ind]
        out_coords[current_ind] += coord[0:2] * ene_current
        for j in range(1, n):
            out_quantities[j, current_ind] += full_quantities[j, quant_ind] * full_quantities[0, quant_ind]
        quant_ind += 1
    if ene_current > 0:
        out_coords[current_ind] /= ene_current
        for j in range(1, n):
            out_quantities[j, current_ind] /= ene_current
        out_mult[current_ind] = n_current
        out_quantities[0, current_ind] = ene_current
    return out_coords, out_quantities, out_mult


@nb.jit(nopython=True)
def calc_arrival_from_peak(fdat, peak_ind):
    peak = fdat[peak_ind]
    thresh = 0.5 * peak
    if peak_ind == 0:
        return 0.5
    cur_ind = peak_ind - 1
    while cur_ind >= 0:
        if fdat[cur_ind] < thresh:
            return cur_ind + 1 + (thresh - fdat[cur_ind]) / (fdat[cur_ind + 1] - fdat[cur_ind])
        elif cur_ind == 0:
            return thresh / fdat[cur_ind]
        cur_ind -= 1
    return 0.


@nb.jit(nopython=True)
def calc_arrival(fdat):
    peak = 0
    for d in fdat:
        if d > peak:
            peak = d
    cur_ind = 0
    thresh = 0.5 * peak
    for d in fdat:
        if d > thresh:
            if cur_ind == 0:
                return cur_ind + thresh / d
            else:
                return cur_ind + (thresh - fdat[cur_ind - 1]) / (d - fdat[cur_ind - 1])
        cur_ind += 1
    return 0.


@nb.jit(nopython=True)
def calc_psd(fdat, arrival_samp, psd_window_lo, psd_window_hi, psd_divider, residual_adjust):
    fast = integrate_lininterp_range(fdat, arrival_samp + psd_window_lo, arrival_samp + psd_divider) + \
           (psd_divider - psd_window_lo + 1) * residual_adjust
    slow = integrate_lininterp_range(fdat, arrival_samp + psd_divider, arrival_samp + psd_window_hi) + \
           (psd_window_hi - psd_divider + 1) * residual_adjust
    if (slow + fast) == 0:
        return 0
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


@nb.jit(nopython=True)
def sum1d(vec):
    sum = 0
    for i in range(vec.shape[0]):
        sum += vec[i]
    return sum


@nb.jit(nopython=True)
def lin_interp_inverse(xy, y):
    """xy is vector of shape (n,2), second index denotes x (0) and y (1)"""
    for i in range(xy.shape[0]):
        if xy[i, 1] > y:
            if i == 0:
                return xy[i, 0]
            else:
                return xy[i - 1, 0] + (y - xy[i - 1, 1]) * ((xy[i, 0] - xy[i - 1, 0]) / (xy[i, 1] - xy[i - 1, 1]))
    return xy[xy.shape[0] - 1, 0]


@nb.jit(nopython=True)
def lin_interp(xy, x):
    """xy is vector of shape (n,2), second index denotes x (0) and y (1)"""
    for i in range(xy.shape[0]):
        if xy[i, 0] > x:
            if i == 0:
                return xy[i, 1]
            else:
                return xy[i - 1, 1] + (x - xy[i - 1, 0]) * ((xy[i, 1] - xy[i - 1, 1]) / (xy[i, 0] - xy[i - 1, 0]))
    return xy[xy.shape[0] - 1, 1]


@nb.jit(nopython=True)
def remove_end_zeros(v, val=0):
    if v[0] == val and val == 0:
        return v[0:1]
    elif v[0] == val:
        return None
    for i in range(1, v.shape[0]):
        if v[i] == val:
            return v[0:i]


@nb.jit(nopython=True)
def find_peaks(v, maxloc, sep):
    local_maxima = zeros(shape=(50,), dtype=int32)
    maxima_vals = zeros(shape=(50,), dtype=float32)
    local_maxpos = 100000
    max_index = 0
    global_maxpos = 0
    for i in range(1, v.shape[0]):
        if v[i] > v[i - 1]:
            local_maxpos = i
        elif v[i] < v[i - 1] and local_maxpos != 100000:
            lmax = int((local_maxpos + i - 1) / 2)
            local_maxima[max_index] = lmax
            maxima_vals[max_index] = v[lmax]
            max_index += 1
            if v[lmax] > v[global_maxpos]:
                global_maxpos = lmax
            if max_index >= 50:
                break
            local_maxpos = 100000
    local_maxima = remove_end_zeros(local_maxima)
    maxima_vals = remove_end_zeros(maxima_vals)
    if local_maxima.shape[0] == 1 and local_maxima[0] == 0 and maxima_vals[0] == 0:
        return 0
    maxima_vals, local_maxima = merge_sort_two(maxima_vals, local_maxima)
    if local_maxima.shape[0] == 1:
        maxloc[0] = local_maxima[0]
        return global_maxpos
    global_maxpos = local_maxima[0]
    maxloc[0] = global_maxpos
    max_index = 1
    for i in range(local_maxima.shape[0] - 1):
        within_range = True
        for j in range(max_index):
            if abs(local_maxima[i + 1] - maxloc[j]) <= sep * 2:
                within_range = False
                break
        if within_range:
            maxloc[max_index] = local_maxima[i + 1]
            max_index += 1
        if max_index > 4:
            break

    """
    current_candidate = local_maxima[0]
    for i in range(1, local_maxima.shape[0]):
        if (local_maxima[i] - current_candidate) <= sep:
            if v[local_maxima[i]] > v[current_candidate]:
                current_candidate = local_maxima[i]
        else:
            maxloc[max_index] = current_candidate
            max_index += 1
            if max_index > 4:
                return global_maxpos
            current_candidate = local_maxima[i]
    """
    return global_maxpos


@nb.jit(nopython=True)
def get_residual(baseline):
    return round(baseline) - baseline


@nb.jit(nopython=True)
def calc_size(data, peak_ind):
    start = peak_ind - 3
    stop = peak_ind + 25
    n = start - stop + 1
    # baseline = find_baseline(data, peak_ind, -30, -5)
    baseline = 0
    residual_adjust = get_residual(baseline)
    return sum_range(data, start, stop) + n * residual_adjust


@nb.jit(nopython=True)
def find_baseline(data, peakloc, baseline_window_lo, baseline_window_hi):
    r_start = peakloc + baseline_window_lo
    r_end = peakloc + baseline_window_hi

    r_start = 0 if r_start < 0 else r_start
    r_end = data.shape[0] if r_end > data.shape[0] else r_end
    if r_end - r_start < 10:
        r_start = 0
        r_end = 10 if 10 < data.shape[0] else data.shape[0]
    return average_median(data[r_start:r_end])


@nb.jit(nopython=True)
def average_median(v, centerfrac=0.33):
    assert v.shape[0] and centerfrac <= 1
    if v.shape[0] == 0:
        return 0
    v = merge_sort_main_numba(v, sort="a")
    res = centerfrac * v.shape[0]
    if 1 > res:
        ndiscard = v.shape[0] - 1
    else:
        ndiscard = v.shape[0] - int(centerfrac * v.shape[0])
    istart = int(ndiscard / 2)
    iend = v.shape[0] - istart
    dsum = 0
    for i in range(istart, iend):
        dsum += v[i]
    return dsum / (iend - istart)


@nb.jit(nopython=True)
def peak_to_dt(wf, m0, m1, x, y, t_interp_curves, sample_times, rel_times, gain_factors,
               sample_width=4, n_samples=150):
    """
    @param wf: waveform for pmt 0 and 1 concatenated (scaled to a 32 bit float between 0 and 1)
    @param m0: sample position of peak for pmt 0
    @param m1: sample position of peak for pmt 1
    @param x: cell x number (0 indexed)
    @param y: cell y number
    @param t_interp_curves:
    @param sample_times: sample time microadjustment
    @param rel_times: relative times for each pmt pair
    @param sample_width: [ns]
    @param n_samples: number of samples in each waveform
    @return: z [mm], E [MeV]
    """
    t = [calc_arrival_from_peak(wf[0:n_samples], m0) * float(sample_width),
         calc_arrival_from_peak(wf[n_samples:], m1) * float(sample_width)]
    for i in range(2):
        if t_interp_curves[x, y, i, 10, 0] == 0:
            continue
        t0 = sample_times[x, y, i] * floor(t[i] / sample_times[x, y, i])
        t[i] = t0 + lin_interp(t_interp_curves[x, y, i], t[i] - t0)
    L = [calc_size(wf[0:n_samples], m0) * gain_factors[x, y, 0],
         calc_size(wf[n_samples:], m1) * gain_factors[x, y, 1]]
    return t[1] - t[0] - rel_times[x, y], L[0] + L[1]


@nb.jit(nopython=True)
def peak_to_z(wf, m0, m1, x, y, gain_factors, t_interp_curves, sample_times, rel_times, eres, light_pos_curves,
              time_pos_curves, light_sum_curves, sample_width=4, n_samples=150):
    """
    @param wf: waveform for pmt 0 and 1 concatenated (scaled to a 32 bit float between 0 and 1)
    @param m0: sample position of peak for pmt 0
    @param m1: sample position of peak for pmt 1
    @param x: cell x number (0 indexed)
    @param y: cell y number
    @param gain_factors: map of gains for each pmt multiplied by 2**14-1
    @param t_interp_curves:
    @param sample_times: sample time microadjustment
    @param rel_times: relative times for each pmt pair
    @param eres: energy resolution of each pmt
    @param light_pos_curves: light ratio position curves
    @param time_pos_curves: dt to position curves
    @param light_sum_curves: light output at each position (for energy correction)
    @param sample_width: [ns]
    @param n_samples: number of samples in each waveform
    @return: z [mm], E [MeV]
    """
    t = [calc_arrival_from_peak(wf[0:n_samples], m0) * float(sample_width),
         calc_arrival_from_peak(wf[n_samples:], m1) * float(sample_width)]
    for i in range(2):
        if t_interp_curves[x, y, i, 10, 0] == 0:
            continue
        t0 = sample_times[x, y, i] * floor(t[i] / sample_times[x, y, i])
        t[i] = t0 + lin_interp(t_interp_curves[x, y, i], t[i] - t0)
    dt = t[1] - t[0] - rel_times[x, y]
    tpos = lin_interp(time_pos_curves[x, y], dt)
    L = [calc_size(wf[0:n_samples], m0) * gain_factors[x, y, 0],
         calc_size(wf[n_samples:], m1) * gain_factors[x, y, 1]]
    if L[0] == 0 or L[1] == 0:
        return 0., (L[0] + L[1]) / lin_interp(light_sum_curves[x, y], 0.)
    PE = [L[0] * eres[x, y, 0], L[1] * eres[x, y, 1]]
    R = log(L[1] / L[0])
    validratio = (R == R)
    dR = sqrt(1.0 / max([PE[0], 1.0]) + 1.0 / max([PE[1], 1.0]))
    # tpos = lin_interp(time_pos_curves[x, y], dt)
    Rpos = lin_interp(light_pos_curves[x, y], R) if validratio else 0
    dRpos = abs(lin_interp(light_pos_curves[x, y], R + 0.5 * dR) - lin_interp(
        light_pos_curves[x, y], R - 0.5 * dR)) if validratio else 0
    Rweight = 1. / (dRpos * dRpos) if (dRpos > 0) else 0
    tweight = 1. / (60 * 60)
    z = (Rweight * Rpos + tweight * tpos) / (Rweight + tweight)
    z = z if abs(z) < 650 else -650. if z < -650 else 650
    E = (PE[0] + PE[1]) / lin_interp(light_sum_curves[x, y], z)
    return z, E


@nb.jit(nopython=True)
def excluded_inds(inds, size):
    if size <= inds.shape[0]:
        print("error, inds must have fewer items than size")
        return None
    duplicates = 0
    if inds.shape[0] > 1:
        # check for duplicates
        for i in range(inds.shape[0] - 1):
            for j in range(i + 1, inds.shape[0]):
                if inds[i] == inds[j]:
                    duplicates += 1
                    break
    exc = zeros((size - inds.shape[0] + duplicates,), dtype=int32)
    cur_ind = 0
    for i in range(size):
        included = False
        for j in inds:
            if i == j:
                included = True
                break
        if not included:
            exc[cur_ind] = i
            cur_ind += 1
    return exc


@nb.jit(nopython=True)
def z_from_total_light(wf, x, y, gain_factors, eres, light_pos_curves,
                       light_sum_curves, n_samples=150):
    L = [sum1d(wf[0:n_samples]) * gain_factors[x, y, 0],
         sum1d(wf[n_samples:]) * gain_factors[x, y, 1]]
    if L[0] == 0 or L[1] == 0:
        return 0., 1. / 100000., (L[0] + L[1]) / lin_interp(light_sum_curves[x, y], 0.)
    PE = [L[0] * eres[x, y, 0], L[1] * eres[x, y, 1]]
    R = log(L[1] / L[0])
    validratio = (R == R)
    z = lin_interp(light_pos_curves[x, y], R) if validratio else 0
    z = z if abs(z) < 650 else -650. if z < -650 else 650
    dR = sqrt(1.0 / max([PE[0], 1.0]) + 1.0 / max([PE[1], 1.0]))
    dRpos = abs(lin_interp(light_pos_curves[x, y], R + 0.5 * dR) - lin_interp(
        light_pos_curves[x, y], R - 0.5 * dR)) if validratio else 0
    Rweight = 1. / (dRpos * dRpos) if (dRpos > 0) else 0
    E = (PE[0] + PE[1]) / lin_interp(light_sum_curves[x, y], z)
    return z, Rweight, E


@nb.jit(nopython=True)
def match_peaks(small, large):
    # dumb way to match peaks that could have duplicates
    ldiffs = zeros((small.shape[0],), dtype=int32)
    for i in range(small.shape[0]):
        ldiffs[i] = 100000
    inds = zeros((small.shape[0],), dtype=int32)
    for i in range(small.shape[0]):
        for j in range(large.shape[0]):
            diff = abs(small[i] - large[j])
            if diff < ldiffs[i]:
                ldiffs[i] = diff
                inds[i] = j
    return inds


@nb.jit(nopython=True)
def z_dt_to_z(wf, z_dt, x, y, gain_factors, eres, light_pos_curves, light_sum_curves, n_samples=150):
    z_dt_weight = 1. / (60. * 60.)
    z_light, z_light_weight, E = z_from_total_light(wf, x, y, gain_factors, eres,
                                                    light_pos_curves, light_sum_curves, n_samples)
    return (z_dt_weight * z_dt + z_light * z_light_weight) / (z_light_weight + z_dt_weight), E


@nb.jit(nopython=True)
def dt_to_z(wf, dt, x, y, gain_factors, eres, light_pos_curves, light_sum_curves, time_pos_curves, n_samples=150):
    z_dt = lin_interp(time_pos_curves[x, y], dt)
    z_dt_weight = 1. / (60. * 60.)
    z_light, z_light_weight, E = z_from_total_light(wf, x, y, gain_factors, eres,
                                                    light_pos_curves, light_sum_curves, n_samples)
    return (z_dt_weight * z_dt + z_light * z_light_weight) / (z_light_weight + z_dt_weight), E


@nb.jit(nopython=True)
def cull_peaks(peaks, culled_peaks, wf, max_loc):
    i = 0
    for p in peaks:
        if p == -1:
            break
        val = wf[p] * MAX_RANGE
        if val > 30 or (wf[p] > 15 and p == max_loc):
            culled_peaks[i] = p
            i += 1


@nb.jit(nopython=True)
def calc_calib_z_E(coordinates, waveforms, z_out, E_out, sample_width, t_interp_curves, sample_times, rel_times,
                   gain_factors,
                   eres, time_pos_curves, light_pos_curves, light_sum_curves, z_scale, n_samples):
    minsep = 10
    for coord, wf in zip(coordinates, waveforms):
        local_maxima0 = zeros((5,), dtype=int32)  # unlikely there would be more than 5
        local_maxima1 = zeros((5,), dtype=int32)
        culled_maxima0 = zeros((5,), dtype=int32)
        culled_maxima1 = zeros((5,), dtype=int32)
        for i in range(5):
            local_maxima0[i] = -1
            local_maxima1[i] = -1
            culled_maxima0[i] = -1
            culled_maxima1[i] = -1
        maxloc0 = find_peaks(wf[0:n_samples], local_maxima0, minsep)
        maxloc1 = find_peaks(wf[n_samples:], local_maxima1, minsep)
        cull_peaks(local_maxima0, culled_maxima0, wf[0:n_samples], maxloc0)
        cull_peaks(local_maxima1, culled_maxima1, wf[n_samples:], maxloc1)
        local_maxima0 = remove_end_zeros(culled_maxima0, -1)
        local_maxima1 = remove_end_zeros(culled_maxima1, -1)
        if local_maxima0 is None or local_maxima1 is None:
            if local_maxima0 is None and local_maxima1 is None:
                continue
            elif local_maxima0 is None:
                r = 1
            else:
                r = 0
            z_out[coord[2], coord[0], coord[1]] = 0.5
            L = sum1d(wf[n_samples*r:n_samples + n_samples*r]) * gain_factors[coord[0], coord[1], r]
            PE = L * eres[coord[0], coord[1], r]
            E_out[coord[2], coord[0], coord[1]] = PE / lin_interp(light_sum_curves[coord[0], coord[1]], 0)
        else:
            if local_maxima0.shape[0] > 1:
                local_maxima0 = merge_sort_main_numba(local_maxima0)
            if local_maxima1.shape[0] > 1:
                local_maxima1 = merge_sort_main_numba(local_maxima1)
            if local_maxima0.shape[0] == local_maxima1.shape[0]:
                z_dt_weighted = 0.
                total_area = 0.
                for m0, m1 in zip(local_maxima0, local_maxima1):
                    peak_z, peak_E = peak_to_z(wf, m0, m1, coord[0], coord[1], gain_factors, t_interp_curves,
                                               sample_times,
                                               rel_times, eres, light_pos_curves, time_pos_curves, light_sum_curves,
                                               sample_width, n_samples)

                    # peak_dt, peak_area = peak_to_dt(wf, m0, m1, coord[0], coord[1], t_interp_curves, sample_times,
                    #                                rel_times, gain_factors, sample_width, n_samples)
                    z_dt_weighted += peak_z * peak_E
                    total_area += peak_E
                z_dt = z_dt_weighted / total_area
                # z, E = z_dt_to_z(wf, z_dt, coord[0], coord[1], gain_factors, eres, light_pos_curves, light_sum_curves,
                #                 n_samples)
                z_out[coord[2], coord[0], coord[1]] = z_dt / z_scale + 0.5
                E_out[coord[2], coord[0], coord[1]] = total_area
            else:
                z_weighted = 0.
                total_E = 0.
                if local_maxima0.shape[0] < local_maxima1.shape[0]:
                    inds = match_peaks(local_maxima0, local_maxima1)
                    no_matches = excluded_inds(inds, local_maxima1.shape[0])
                    for i in range(local_maxima0.shape[0]):
                        peak_dt, peak_area = peak_to_dt(wf, local_maxima0[i], local_maxima1[inds[i]], coord[0],
                                                        coord[1], t_interp_curves, sample_times,
                                                        rel_times, gain_factors, sample_width, n_samples)
                        # peak_z, peak_E = peak_to_z(wf, local_maxima0[i], local_maxima1[inds[i]], coord[0], coord[1],
                        #                           gain_factors, t_interp_curves, sample_times, rel_times, eres,
                        #                           light_pos_curves, time_pos_curves, light_sum_curves, sample_width,
                        #                           n_samples)
                        z_weighted += peak_dt * peak_area
                        total_E += peak_area
                else:
                    inds = match_peaks(local_maxima1, local_maxima0)
                    no_matches = excluded_inds(inds, local_maxima0.shape[0])
                    for i in range(local_maxima1.shape[0]):
                        peak_dt, peak_area = peak_to_dt(wf, local_maxima0[inds[i]], local_maxima1[i], coord[0],
                                                        coord[1], t_interp_curves, sample_times,
                                                        rel_times, gain_factors, sample_width, n_samples)
                        # peak_z, peak_E = peak_to_z(wf, local_maxima0[inds[i]], local_maxima1[i], coord[0], coord[1],
                        #                           gain_factors, t_interp_curves, sample_times, rel_times, eres,
                        #                           light_pos_curves, time_pos_curves, light_sum_curves, sample_width,
                        #                           n_samples)
                        z_weighted += peak_dt * peak_area
                        total_E += peak_area
                z_dt = z_weighted / total_E
                z, E = z_dt_to_z(wf, z_dt, coord[0], coord[1], gain_factors, eres, light_pos_curves, light_sum_curves,
                                 n_samples)
                z_out[coord[2], coord[0], coord[1]] = z / z_scale + 0.5
                E_out[coord[2], coord[0], coord[1]] = E


@nb.jit(nopython=True)
def E_basic_prediction_dense(E, z, nx, ny, seg_status, light_pos_curves, light_sum_curves, pred):
    """assumes z contains some z prediction for single ended"""
    """E is dense matrix, first index is batch index, second index is feature index,
    features are energy, PE0, PE1 in MeV and photons 
    z is dense matrix first index is batch index, second and third indices are cooirdinate"""
    for batch in range(E.shape[0]):
        for x in range(nx):
            for y in range(ny):
                if E[batch, 0, x, y] == 0:
                    continue
                if seg_status[x, y] > 0:
                    if E[batch, 1, x, y] == 0 and E[batch, 2, x, y] == 0:
                        continue
                    elif E[batch, 1, x, y] != 0 and E[batch, 2, x, y] != 0:
                        print("error: seg status is incongruent with PE0, PE1, for segment ")
                        print(x)
                        print(y)
                    logR = lin_interp_inverse(light_pos_curves[x, y], z[batch, x, y])
                    if E[batch, 1, x, y] == 0:
                        P0 = E[batch, 2, x, y] / exp(logR)
                        pred[batch, x, y] = (P0 + E[batch, 2, x, y]) / lin_interp(light_sum_curves[x, y],
                                                                                  z[batch, x, y])
                    else:
                        P1 = E[batch, 1, x, y] * exp(logR)
                        pred[batch, x, y] = (E[batch, 1, x, y] + P1) / lin_interp(light_sum_curves[x, y],
                                                                                  z[batch, x, y])
                else:
                    pred[batch, x, y] = E[batch, 0, x, y]


@nb.jit(nopython=True)
def E_basic_prediction(coo, E, PE0, PE1, z, seg_status, light_pos_curves, light_sum_curves, pred):
    """assumes z contains some z prediction for single ended"""
    for batch in range(coo.shape[0]):
        x = coo[batch, 0]
        y = coo[batch, 1]
        if seg_status[x, y] > 0:
            if PE0[batch] == 0 and PE1[batch] == 0:
                continue
            elif PE0[batch] != 0 and PE1[batch] != 0:
                print("error: seg status is incongruent with PE0, PE1, for segment ")
            logR = lin_interp_inverse(light_pos_curves[x, y], z[batch])
            if PE0[batch] == 0:
                P0 = PE1[batch] / exp(logR)
                pred[batch] = (P0 + PE1[batch]) / lin_interp(light_sum_curves[x, y], z[batch])
            else:
                P1 = PE0[batch] * exp(logR)
                pred[batch] = (PE0[batch] + P1) / lin_interp(light_sum_curves[x, y], z[batch])
        else:
            pred[batch] = E[batch]


@nb.jit(nopython=True)
def z_basic_prediction_dense(z_pred, z_truth, truth_is_cal=False):
    for i in range(z_pred.shape[0]):
        for x in range(z_pred.shape[1]):
            for y in range(z_pred.shape[2]):
                if z_truth[i, x, y] != 0 and z_pred[i, x, y] == 0.5:
                    sum = 0.0
                    n = 0
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            if j == k == 0:
                                continue
                            if 0 <= x + j < z_pred.shape[1] and 0 <= y + k < z_pred.shape[2]:
                                if truth_is_cal:
                                    z_pred[i, x + j, y + k] = z_truth[i, x + j, y + k]
                                    if z_pred[i, x + j, y + k] != 0 and z_pred[i, x, y] != 0.5:
                                        sum += z_pred[i, x + j, y + k]
                                        n += 1
                                else:
                                    if z_pred[i, x + j, y + k] != 0 and z_pred[i, x, y] != 0.5:
                                        sum += z_pred[i, x + j, y + k]
                                        n += 1
                    if n > 0:
                        z_pred[i, x, y] = sum / n


@nb.jit(nopython=True)
def z_basic_prediction(coo, feat, pred):
    cur_ind = coo[0, 2]
    for i in range(coo.shape[0]):
        if coo[i, 2] != cur_ind:
            cur_ind = coo[i, 2]
        if feat[i] != 0.5:
            pred[i] = feat[i]
        else:
            j = i - 1
            p = 0.0
            n = 0
            while coo[j, 2] == cur_ind:
                if abs(coo[j, 0] - coo[i, 0]) <= 1 and abs(coo[j, 1] - coo[i, 1]) <= 1:
                    if feat[j] != 0.5:
                        p += feat[j]
                        n += 1
                j -= 1
            j = i + 1
            while coo[j, 2] == cur_ind:
                if abs(coo[j, 0] - coo[i, 0]) <= 1 and abs(coo[j, 1] - coo[i, 1]) <= 1:
                    if feat[j] != 0.5:
                        p += feat[j]
                        n += 1
                j += 1
            if n > 0:
                pred[i] = p / n
            else:
                pred[i] = 0.5


@nb.jit(nopython=True)
def increment_metric_mult_SE(dev, bin_number, i, j, mult, nmult, out_dev, out_n, single_dev, single_n, dual_dev,
                             dual_n, seg_status):
    if 0 < mult <= nmult:
        out_dev[i, j, mult - 1] += dev
        out_n[i, j, mult - 1] += 1
        if seg_status[i, j] > 0:
            single_dev[bin_number, mult - 1] += dev
            single_n[bin_number, mult - 1] += 1
        else:
            dual_dev[bin_number, mult - 1] += dev
            dual_n[bin_number, mult - 1] += 1
    else:
        out_dev[i, j, nmult] += dev
        out_n[i, j, nmult] += 1
        if seg_status[i, j] > 0:
            single_dev[bin_number, nmult] += dev
            single_n[bin_number, nmult] += 1
        else:
            dual_dev[bin_number, nmult] += dev
            dual_n[bin_number, nmult] += 1


@nb.jit(nopython=True)
def increment_metric_SE_2d(dev, bin_x, bin_y, i, j, single_dev, single_n, dual_dev,
                             dual_n, seg_status):
    if seg_status[i, j] > 0:
        single_dev[bin_x, bin_y] += dev
        single_n[bin_x, bin_y] += 1
    else:
        dual_dev[bin_x, bin_y] += dev
        dual_n[bin_x, bin_y] += 1

@nb.jit(nopython=True)
def E_deviation(predictions, targets, dev, out_n, E_mult_dual_dev, E_mult_dual_out, E_mult_single_dev,
                E_mult_single_out, seg_status, nx, ny, nmult, nE, E_low, E_high, E_scale):
    bin_width = (E_high - E_low) / nE
    for batch in range(predictions.shape[0]):
        mult = 0
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    mult += 1
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    E_dev = abs(predictions[batch, i, j] - targets[batch, i, j]) / targets[batch, i, j]
                    true_E = targets[batch, i, j] * E_scale
                    E_bin = get_bin_index(true_E, E_low, E_high, bin_width, nE)
                    increment_metric_mult_SE(E_dev, E_bin, i, j, mult, nmult, dev, out_n, E_mult_single_dev,
                                             E_mult_single_out, E_mult_dual_dev, E_mult_dual_out, seg_status)


@nb.jit(nopython=True)
def E_deviation_with_z(predictions, targets, dev, out_n, E_mult_dual_dev, E_mult_dual_out, E_mult_single_dev,
                       E_mult_single_out, seg_status, nx, ny, nmult, nE, E_low, E_high, E_scale, zrange, Z,
                       E_z_dual_dev, E_z_dual_out, E_z_single_dev, E_z_single_out):
    E_bin_width = (E_high - E_low) / nE
    for batch in range(predictions.shape[0]):
        mult = 0
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    mult += 1
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    E_dev = abs(predictions[batch, i, j] - targets[batch, i, j]) / targets[batch, i, j]
                    true_E = targets[batch, i, j] * E_scale
                    E_bin = get_bin_index(true_E, E_low, E_high, E_bin_width, nE)
                    z_bin = get_bin_index((Z[batch, i, j] - 0.5) * zrange, -zrange / 2., zrange / 2., zrange / nE, nE)
                    if 0 < mult <= nmult:
                        dev[i, j, mult - 1] += E_dev
                        out_n[i, j, mult - 1] += 1
                        if seg_status[i, j] > 0:
                            E_mult_single_dev[E_bin, mult - 1] += E_dev
                            E_mult_single_out[E_bin, mult - 1] += 1
                            E_z_single_dev[E_bin, z_bin] += E_dev
                            E_z_single_out[E_bin, z_bin] += 1
                        else:
                            E_mult_dual_dev[E_bin, mult - 1] += E_dev
                            E_mult_dual_out[E_bin, mult - 1] += 1
                            E_z_dual_dev[E_bin, z_bin] += E_dev
                            E_z_dual_out[E_bin, z_bin] += 1
                    else:
                        dev[i, j, nmult] += E_dev
                        out_n[i, j, nmult] += 1
                        if seg_status[i, j] > 0:
                            E_mult_single_dev[E_bin, nmult] += E_dev
                            E_mult_single_out[E_bin, nmult] += 1
                            E_z_single_dev[E_bin, z_bin] += E_dev
                            E_z_single_out[E_bin, z_bin] += 1
                        else:
                            E_mult_dual_dev[E_bin, nmult] += E_dev
                            E_mult_dual_out[E_bin, nmult] += 1
                            E_z_dual_dev[E_bin, z_bin] += E_dev
                            E_z_dual_out[E_bin, z_bin] += 1


@nb.jit(nopython=True)
def z_deviation(predictions, targets, dev, out_n, z_mult_dual_dev, z_mult_dual_out, z_mult_single_dev,
                z_mult_single_out, seg_status, nx, ny, nmult, nz, zrange):
    for batch in range(predictions.shape[0]):
        mult = 0
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    mult += 1
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    z_dev = abs(predictions[batch, i, j] - targets[batch, i, j])
                    true_z = (targets[batch, i, j] - 0.5) * zrange
                    z_bin = 0
                    if true_z < (-zrange / 2.):
                        z_bin = 0
                    elif true_z >= (zrange / 2.):
                        z_bin = nz + 1
                    else:
                        for k in range(1, nz + 1):
                            if k * (zrange / nz) - zrange / 2. > true_z:
                                z_bin = k
                                break
                    increment_metric_mult_SE(z_dev, z_bin, i, j, mult, nmult, dev, out_n, z_mult_single_dev,
                                             z_mult_single_out, z_mult_dual_dev, z_mult_dual_out, seg_status)


@nb.jit(nopython=True)
def z_deviation_with_E_full_correlation(predictions, targets, dev, out_n, z_mult_dual_dev, z_mult_dual_out, z_mult_single_dev,
                z_mult_single_out, z_E_single_dev, z_E_single_out, z_E_dual_dev, z_E_dual_out, E_mult_single_dev,
                                        E_mult_single_out, E_mult_dual_dev, E_mult_dual_out, seg_status, blindl, nx, ny, nmult,
                                        nz, zrange, E, E_low, E_high, nE):
    E_bin_width = (E_high - E_low) / nE
    z_bin_width = zrange / nz
    half_cell_length = 588.
    for batch in range(predictions.shape[0]):
        mult = 0
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    mult += 1
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    z_dev = abs(predictions[batch, i, j] - targets[batch, i, j])
                    true_z = (targets[batch, i, j] - 0.5) * zrange
                    z_bin = 0
                    E_bin = get_bin_index(E[batch, i, j], E_low, E_high, E_bin_width, nE)
                    if seg_status[i, j] == 0.5:
                        if blindl[i, j] == 1:
                            dist_pmt = half_cell_length - true_z
                        else:
                            dist_pmt = half_cell_length + true_z
                        z_bin = get_bin_index(dist_pmt, 0., zrange, z_bin_width, nz)
                        increment_metric_mult_SE(z_dev, z_bin, i, j, mult, nmult, dev, out_n, z_mult_single_dev,
                                                 z_mult_single_out, z_mult_dual_dev, z_mult_dual_out, seg_status)
                        increment_metric_SE_2d(z_dev, z_bin, E_bin, i, j, z_E_single_dev, z_E_single_out, z_E_dual_dev,
                                               z_E_dual_out, seg_status)
                    elif seg_status[i, j] == 0:
                        dist_pmt = half_cell_length + true_z
                        z_bin = get_bin_index(dist_pmt, 0., zrange, z_bin_width, nz)
                        increment_metric_mult_SE(z_dev, z_bin, i, j, mult, nmult, dev, out_n, z_mult_single_dev,
                                                 z_mult_single_out, z_mult_dual_dev, z_mult_dual_out, seg_status)
                        increment_metric_SE_2d(z_dev, z_bin, E_bin, i, j, z_E_single_dev, z_E_single_out, z_E_dual_dev,
                                               z_E_dual_out, seg_status)
                        dist_pmt = half_cell_length - true_z
                        z_bin = get_bin_index(dist_pmt, 0., zrange, z_bin_width, nz)
                        increment_metric_mult_SE(z_dev, z_bin, i, j, mult, nmult, dev, out_n, z_mult_single_dev,
                                                 z_mult_single_out, z_mult_dual_dev, z_mult_dual_out, seg_status)
                        increment_metric_SE_2d(z_dev, z_bin, E_bin, i, j, z_E_single_dev, z_E_single_out, z_E_dual_dev,
                                               z_E_dual_out, seg_status)
                    if(mult > nmult):
                        mult_bin = nmult
                    else:
                        mult_bin = mult-1
                    increment_metric_SE_2d(z_dev, E_bin, mult_bin, i, j, E_mult_single_dev, E_mult_single_out, E_mult_dual_dev,
                                           E_mult_dual_out, seg_status)

@nb.jit(nopython=True)
def z_deviation_with_E(predictions, targets, dev, out_n, z_mult_dual_dev, z_mult_dual_out, z_mult_single_dev,
                       z_mult_single_out, seg_status, nx, ny, nmult, nz, zrange, E,
                       E_mult_dual_dev, E_mult_dual_out, E_mult_single_dev, E_mult_single_out,
                       E_low, E_high):
    E_bin_width = (E_high - E_low) / nz
    for batch in range(predictions.shape[0]):
        mult = 0
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    mult += 1
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    z_dev = abs(predictions[batch, i, j] - targets[batch, i, j])
                    true_z = (targets[batch, i, j] - 0.5) * zrange
                    E_bin = get_bin_index(E[batch, i, j], E_low, E_high, E_bin_width, nz)
                    z_bin = 0
                    if true_z < (-zrange / 2.):
                        z_bin = 0
                    elif true_z >= (zrange / 2.):
                        z_bin = nz + 1
                    else:
                        for k in range(1, nz + 1):
                            if k * (zrange / nz) - zrange / 2. > true_z:
                                z_bin = k
                                break
                    if 0 < mult <= nmult:
                        dev[i, j, mult - 1] += z_dev
                        out_n[i, j, mult - 1] += 1
                        if seg_status[i, j] > 0:
                            z_mult_single_dev[z_bin, mult - 1] += z_dev
                            z_mult_single_out[z_bin, mult - 1] += 1
                            E_mult_single_dev[E_bin, mult - 1] += z_dev
                            E_mult_single_out[E_bin, mult - 1] += 1
                        else:
                            z_mult_dual_dev[z_bin, mult - 1] += z_dev
                            z_mult_dual_out[z_bin, mult - 1] += 1
                            E_mult_dual_dev[E_bin, mult - 1] += z_dev
                            E_mult_dual_out[E_bin, mult - 1] += 1
                    else:
                        dev[i, j, nmult] += z_dev
                        out_n[i, j, nmult] += 1
                        if seg_status[i, j] > 0:
                            z_mult_single_dev[z_bin, nmult] += z_dev
                            z_mult_single_out[z_bin, nmult] += 1
                            E_mult_single_dev[E_bin, nmult] += z_dev
                            E_mult_single_out[E_bin, nmult] += 1
                        else:
                            z_mult_dual_dev[z_bin, nmult] += z_dev
                            z_mult_dual_out[z_bin, nmult] += 1
                            E_mult_dual_dev[E_bin, nmult] += z_dev
                            E_mult_dual_out[E_bin, nmult] += 1


@nb.jit(nopython=True)
def is_in_sample(sample_segs, i, j):
    for seg in sample_segs:
        if i == seg[0] and j == seg[1]:
            return True
    return False


@nb.jit(nopython=True)
def sample_index(sample_segs, i, j):
    for k, seg in enumerate(sample_segs):
        if i == seg[0] and j == seg[1]:
            return k
    return -1


@nb.jit(nopython=True)
def z_error(predictions, targets, results, n_bins, low, high, nmult, sample_segs, zrange):
    bin_width = (high - low) / n_bins
    for batch in range(predictions.shape[0]):
        mult = 0
        has_sample = False
        for i in range(targets.shape[1]):
            for j in range(targets.shape[2]):
                if targets[batch, i, j] > 0:
                    if is_in_sample(sample_segs, i, j):
                        has_sample = True
                    mult += 1
        if not has_sample:
            continue
        for i in range(targets.shape[1]):
            for j in range(targets.shape[2]):
                if targets[batch, i, j] > 0:
                    if not is_in_sample(sample_segs, i, j):
                        continue
                    s_ind = sample_index(sample_segs, i, j)
                    z_err = (predictions[batch, i, j] - targets[batch, i, j]) * zrange
                    err_bin = 0
                    if z_err < low:
                        err_bin = 0
                    elif z_err >= high:
                        err_bin = n_bins + 1
                    else:
                        for k in range(1, n_bins + 1):
                            if k * bin_width + low > z_err:
                                err_bin = k
                                break
                    if 0 < mult <= nmult:
                        results[s_ind, mult - 1, err_bin] += 1
                    else:
                        results[s_ind, nmult, err_bin] += 1
