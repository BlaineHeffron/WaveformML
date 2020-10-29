from scipy import stats
import numpy as np

def moment_prod(x, counts):
    return np.sum(counts*x[None,:],axis=1) / np.sum(counts, axis=1)

def calc_moments(dist_vec, n):
    """dist_vec is the vector of distributions, batch index is given by the first index, 1-d distribution along second dimension"""
    mult_values = []
    output = np.zeros((dist_vec.shape[0], n))
    pulses = dist_vec[:,:150] + dist_vec[:,150:]
    for i in range(n):
        output[:,i] = moment_prod(np.arange(2,150*4+2,4)**(i+2),pulses)
    return output
