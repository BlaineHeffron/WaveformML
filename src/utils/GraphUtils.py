import torch
from math import ceil
from src.custom_functions import lib, ffi
import time


def window_edges(coo, batch, max_dist=1, self_loops=True):
    """
    Calculate edges based on window size of a convolutional filter
    :param coo: coordinate tensor
    :type coo: torch.IntTensor
    :return: edge index
    :rtype: torch.LongTensor
    """

    #print(coo.dtype)
    assert (coo.dtype == torch.int64)

    # Initialize output tensor to hold the edge indices
    coolen = coo.shape[0]
    edge_index = torch.zeros((2, coolen * int(ceil(coolen / (batch[-1] - batch[0])))*2), dtype=torch.int64)
    if not edge_index.is_contiguous():
        edge_index = edge_index.contiguous()
    coo = coo.transpose(0, 1).contiguous()
    if not batch.is_contiguous():
        batch = batch.contiguous()
    cur_index = torch.tensor([0], dtype=torch.int64)
    # For CFFI, the inputs have to be cast to the target C equivalents using cffi.ffi.cast.
    # Afterwards, the C function can be called like a regular Python function using the converted arguments.
    _w = ffi.cast('long long', max_dist + 1)
    _c = ffi.cast('long long*', cur_index.data_ptr())
    _n = ffi.cast('long long', coolen)
    _x = ffi.cast('long long*', coo[0].data_ptr())
    _y = ffi.cast('long long*', coo[1].data_ptr())
    _b = ffi.cast('long long*', batch.data_ptr())
    _sl = ffi.cast('bool', self_loops)
    _edge1 = ffi.cast('long long*', edge_index[0].data_ptr())
    _edge2 = ffi.cast('long long*', edge_index[1].data_ptr())
    lib.cffi_window_edges(_w, _c, _n, _x, _y, _b, _sl, _edge1, _edge2)
    return edge_index[:, 0:cur_index[0]]

def get_edges(n, c):
    edges0 = []
    edges1 = []
    for i in range(c.shape[0]):
        j = i + 1
        edges0.append(i)
        edges1.append(i)
        while j < c.shape[0] and c[i,2] == c[j,2]:
            if abs(c[i, 0] - c[j, 0]) <= n and abs(c[i, 1] - c[j, 1]) <= n:
                edges0.append(i)
                edges1.append(j)
                edges1.append(i)
                edges0.append(j)
            j += 1
    return torch.stack((torch.tensor(edges0,dtype=torch.int64), torch.tensor(edges1,dtype=torch.int64)), 0)

def test():
    x = torch.randint(0, 4, (100000,), dtype=torch.int64)
    y = torch.randint(0, 4, (100000,), dtype=torch.int64)
    from math import floor
    b = torch.tensor([int(floor(i / 4)) for i in range(100000)], dtype=torch.int64)
    coo = torch.stack((x, y, b), dim=1)
    st = time.time()
    edge_index = window_edges(coo[:, 0:2], coo[:, 2], 3, True)
    #edge_index_compare = get_edges(1, coo)
    print(edge_index.shape)
    #print(edge_index_compare.shape)
    et = time.time()
    print(et - st)

