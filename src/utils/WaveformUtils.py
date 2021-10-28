import numba as nb


@nb.jit(nopython=True)
def align_wfs(data, out, n_before=1):
    """
    @param data: numpy array of shape (N, 2, L) where N is batch size, L is length of waveform, 2 for left and right waveforms
    @param out: numpy array of shape (N, 2, L2) where L2 is length of output waveform
    @param n_before: number of samples before arrival sample to keep
    @return: null
    """
    for i in range(data.shape[0]):
        for j in range(2):
            maxloc = find_peak(data[i, j])
            arrival_samp = maxloc + calc_crossing(data[i, j], -0.5, maxloc)
            print(data[i, j][5:15])
            print(maxloc)
            print(arrival_samp)
            start = int(round(arrival_samp)) - n_before
            if start < 0:
                zero_pad = -1 * start
                start = 0
            else:
                zero_pad = 0
            for k in range(start, data.shape[2]):
                if k + zero_pad < out.shape[2]:
                    out[i, j, k + zero_pad] = data[i, j, k]
                else:
                    break


@nb.jit(nopython=True)
def find_peak(v):
    local_maxpos = 100000
    global_maxpos = 0
    for i in range(1, v.shape[0]):
        if v[i] > v[i - 1]:
            local_maxpos = i
        elif v[i] < v[i - 1] and local_maxpos != 100000:
            lmax = int((local_maxpos + i - 1) / 2)
            if v[lmax] > v[global_maxpos]:
                global_maxpos = lmax
            local_maxpos = 100000
    return global_maxpos


@nb.jit(nopython=True)
def calc_crossing(data, thresh, maxloc):
    """
    thresh = -0.2, -0.5, -0.8, 0.8, 0.2
    -0.5 is for arrival sample
    """
    rising = thresh < 0
    if rising:
        datend_ind = 0
    else:
        datend_ind = data.shape[0]
    hmax = data[maxloc]
    if rising:
        tx = -1 * find_edge_crossing(data, maxloc, datend_ind, abs(thresh) * hmax)
    else:
        tx = find_edge_crossing(data, maxloc, datend_ind, abs(thresh) * hmax)
    if not (0 <= maxloc + tx < data.shape[0]):
        tx = 0
    return tx


@nb.jit(nopython=True)
def find_edge_crossing(data, start_ind, stop_ind, thresh):
    idx = 0
    prevVal = data[start_ind]
    while start_ind != stop_ind:
        if data[start_ind] < thresh:
            break
        prevVal = data[start_ind]
        idx += 1
        if start_ind < stop_ind:
            start_ind += 1
        else:
            start_ind -= 1
    if start_ind == stop_ind:
        return idx
    return idx - 1 + (prevVal - thresh) / (prevVal - data[start_ind])


@nb.jit(nopython=True)
def peak_interpolate(data, maxloc):
    """
    @param data: samples to interpolate
    @param maxloc: sample number of maximum
    @return: tuple of interpolated peak position, heigh
    """
    if maxloc < 1 or maxloc >= data.shape[0] - 1:
        if maxloc < data.shape[0]:
            return maxloc, data[maxloc]
        else:
            return maxloc, 0
    sp = data[maxloc + 1]
    s0 = data[maxloc]
    sm = data[maxloc - 1]
    peakpos = maxloc
    height = s0
    d = 4 * s0 - 2 * sp - 2 * sm
    if d > 1:
        c = (sp - sm) / d
        peakpos += c
        height += (sp - sm) * c / 4
    return peakpos, height
