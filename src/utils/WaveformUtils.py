import numba as nb


@nb.jit(nopython=True)
def align_wfs(data, out):
    n = out.shape[2]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            counter = 0
            start_counter = False
            for k in range(data.shape[2]):
                if data[i,j,k] > 0:
                    start_counter = True
                    out[i,j,counter] = data[i,j,k]
                    counter+=1
                elif start_counter:
                    counter += 1
                if counter == n:
                    break



