import numba as nb
from math import ceil, floor, sqrt, log
from numba.typed import List


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
    # expects output to be of size nbins+2, 1 for overflow and underflow
    xlen = xrange[1] - xrange[0]
    bin_width = xlen / nbins
    for i in range(metric.shape[0]):
        bin = 0
        find_bin = True
        if metric[i] < xrange[0]:
            find_bin = False
        elif metric[i] > xrange[1]:
            bin = nbins + 1
            find_bin = False
        if find_bin:
            for j in range(1, nbins + 1):
                if j * bin_width + xrange[0] > metric[i]:
                    bin = j - 1
                    break
            output[bin, label[i], prediction[i]] += 1


@nb.jit(nopython=True)
def metric_accumulate_1d(results, metric, output, out_n, xrange, nbins):
    # expects output to be of size nbins+2, 1 for overflow and underflow
    xlen = xrange[1] - xrange[0]
    bin_width = xlen / nbins
    for i in range(results.shape[0]):
        bin = 0
        find_bin = True
        if metric[i] < xrange[0]:
            find_bin = False
        elif metric[i] > xrange[1]:
            bin = nbins + 1
            find_bin = False
        if find_bin:
            for j in range(1, nbins + 1):
                if j * bin_width + xrange[0] > metric[i]:
                    bin = j
                    break
        output[bin] += results[i]
        out_n[bin] += 1


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
        elif metric[i, 0] > xrange[1]:
            binx = nbinsx + 1
            find_binx = False
        if metric[i, 1] < yrange[0]:
            find_biny = False
        elif metric[i, 1] > yrange[1]:
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
    sum = 0.0
    for i in range(nsamp):
        t += pulse[i] * (i + 0.5)
        sum += pulse[i]
    if sum != 0.0:
        return t / sum
    else:
        return 0


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
    n_samples = 150
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
        pulses[pulse_ind, 0:n_samples] = pulseleft
        pulses[pulse_ind, n_samples:2 * n_samples] = pulseright
        tot_l = vec_sum(pulseleft)
        tot_r = vec_sum(pulseright)
        tot_l_current += tot_l
        tot_r_current += tot_r
        psdl[current_ind] += calc_psd(pulseleft, calc_arrival(pulseleft),
                                      psd_window_lo, psd_window_hi, psd_divider) * tot_l
        psdr[current_ind] += calc_psd(pulseright, calc_arrival(pulseright),
                                      psd_window_lo, psd_window_hi, psd_divider) * tot_r
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
        for j in range(1, n - 1):
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
def calc_calib_z(coordinates, waveforms, z_out, sample_width, t_interp_curves, sample_times, rel_times, gain_factors,
                 eres, time_pos_curves, light_pos_curves, z_scale):
    i = 0
    for coord, wf in zip(coordinates, waveforms):
        t = [calc_arrival(wf[:, 0]) * sample_width, calc_arrival(wf[:, 1]) * sample_width]
        for i in range(2):
            if t_interp_curves[coord[0], coord[1], i, 10, 0] == 0:
                continue
            t0 = sample_times[coord[0], coord[1], i] * floor(t[i] / sample_times[coord[0], coord[1], i])
            t[i] = t0 + lin_interp(t_interp_curves[coord[0], coord[1], i], t[i] - t0)
        dt = t[1] - t[0] - rel_times[coord[0], coord[1]]
        L = [sum1d(wf[0]) * gain_factors[coord[0], coord[1], 0], sum1d(wf[1]) * gain_factors[coord[0], coord[1], 1]]
        if L[0] == 0 or L[1] == 0:
            z_out[i, coord[0], coord[1]] = 0.5
            i += 1
            continue
        PE = [L[0] * eres[coord[0], coord[1], 0], L[1] * eres[coord[0], coord[1], 1]]
        R = log(L[1] / L[0])
        validratio = (R == R)
        dR = sqrt(1.0 / max([PE[0], 1.0]) + 1.0 / max([PE[1], 1.0]))
        tpos = lin_interp(time_pos_curves[coord[0], coord[1]], dt)
        Rpos = lin_interp(light_pos_curves[coord[0], coord[1]], R) if validratio else 0
        dRpos = abs(lin_interp(light_pos_curves[coord[0], coord[1]], R + 0.5 * dR) - lin_interp(
            light_pos_curves[coord[0], coord[1]], R - 0.5 * dR)) if validratio else 0
        Rweight = 1. / (dRpos * dRpos) if (dRpos > 0) else 0
        tweight = 1. / (60 * 60)
        z_out[i, coord[0], coord[1]] = ((Rweight * Rpos + tweight * tpos) / (Rweight + tweight))/z_scale + 0.5
        i += 1


@nb.jit(nopython=True)
def z_deviation(predictions, targets, dev, out_n, nx, ny, nmult):
    for batch in range(predictions.shape[0]):
        mult = 0
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    mult += 1
        for i in range(nx):
            for j in range(ny):
                if targets[batch, i, j] > 0:
                    if 0 < mult <= nmult:
                        dev[i, j, mult - 1] += abs(predictions[batch, i, j] - targets[batch, i, j])
                        out_n[i, j, mult - 1] += 1
                    else:
                        dev[i, j, nmult] += abs(predictions[batch, i, j] - targets[batch, i, j])
                        out_n[i, j, nmult] += 1
