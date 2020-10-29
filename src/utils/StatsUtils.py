from scipy import stats
import numpy as np


def calc_moments(dist_vec, n):
    """dist_vec is the vector of distributions, batch index is given by the first index, 1-d distribution along second dimension"""
    output = np.zeros((dist_vec.shape[0], n))
    for i in range(n):
        output[:,i] = stats.moment(dist_vec, moment=i + 1, axis=1)
    return output
