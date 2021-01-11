import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from numpy import int32, zeros, array
from src.utils.SparseUtils import find_peaks


def main():
    vec = array([1,2,3,4,5,3,4,3,2,3,1,1,1,4,4,4,3,2,6,3,4,5,6,4,3,4,5,3,2,1,1,2,1,2,1,4,5,3,2,1,1,1,2,2,4,4,2,3,4,5,6,7,5,4,3,4,5,4,3])
    local_max = zeros((5,), dtype=int32)
    global_max = find_peaks(vec,local_max,10)
    print(local_max)
    if local_max[0] == 0:
        local_max = local_max[1:]
    else:
        for i in range(1,local_max.shape[0]):
            if local_max[i] == 0:
                local_max = local_max[0:i]
                break

    print("global max is {0} at {1}".format(vec[global_max],global_max))
    print("local maxes is {0} at {1}".format(vec[local_max],local_max))


if __name__ == "__main__":
    main()
