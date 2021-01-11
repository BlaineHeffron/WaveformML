from numba import jit
import numpy as np

SMALL_MERGESORT_NUMBA = 40


@jit(nopython=True)
def merge_numba(A, Aux, lo, mid, hi):
    i = lo
    j = mid + 1
    for k in range(lo, hi + 1):
        if i > mid:
            Aux[k] = A[j]
            j += 1
        elif j > hi:
            Aux[k] = A[i]
            i += 1
        elif A[j] < A[i]:
            Aux[k] = A[j]
            j += 1
        else:
            Aux[k] = A[i]
            i += 1


@jit(nopython=True)
def merge_numba_two(A, Aux, lo, mid, hi, C, CAux):
    i = lo
    j = mid + 1
    for k in range(lo, hi + 1):
        if i > mid:
            Aux[k] = A[j]
            CAux[k] = C[j]
            j += 1
        elif j > hi:
            Aux[k] = A[i]
            CAux[k] = C[i]
            i += 1
        elif A[j] < A[i]:
            Aux[k] = A[j]
            CAux[k] = C[j]
            j += 1
        else:
            Aux[k] = A[i]
            CAux[k] = C[i]
            i += 1

@jit(nopython=True)
def insertion_sort_numba(A, lo, hi):
    for i in range(lo + 1, hi + 1):
        key = A[i]
        j = i - 1
        while (j >= lo) & (A[j] < key):
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = key


@jit(nopython=True)
def insertion_sort_numba_two(A, lo, hi, C):
    for i in range(lo + 1, hi + 1):
        key = A[i]
        second = C[i]
        j = i - 1
        while (j >= lo) & (A[j] < key):
            A[j + 1] = A[j]
            C[j + 1] = C[j]
            j -= 1
        A[j + 1] = key
        C[j + 1] = second

@jit(nopython=True)
def merge_sort_numba(A, Aux, lo, hi):
    if hi - lo > SMALL_MERGESORT_NUMBA:
        mid = lo + ((hi - lo) >> 1)
        merge_sort_numba(Aux, A, lo, mid)
        merge_sort_numba(Aux, A, mid + 1, hi)
        if A[mid] > A[mid + 1]:
            merge_numba(A, Aux, lo, mid, hi)
        else:
            for i in range(lo, hi + 1):
                Aux[i] = A[i]
    else:
        insertion_sort_numba(Aux, lo, hi)

@jit(nopython=True)
def merge_sort_numba_two(A, Aux, lo, hi, C, CAux):
    if hi - lo > SMALL_MERGESORT_NUMBA:
        mid = lo + ((hi - lo) >> 1)
        merge_sort_numba_two(Aux, A, lo, mid, C, CAux)
        merge_sort_numba_two(Aux, A, mid + 1, hi, C, CAux)
        if A[mid] > A[mid + 1]:
            merge_numba_two(A, Aux, lo, mid, hi, C, CAux)
        else:
            for i in range(lo, hi + 1):
                Aux[i] = A[i]
                CAux[i] = C[i]
    else:
        insertion_sort_numba_two(Aux, lo, hi, CAux)

@jit(nopython=True)
def merge_sort_main_numba(A):
    B = np.copy(A)
    Aux = np.copy(A)
    merge_sort_numba(Aux, B, 0, len(B) - 1)
    return B

@jit(nopython=True)
def merge_sort_two(A, C):
    #sorts C based on A
    B = np.copy(A)
    Aux = np.copy(A)
    second = np.copy(C)
    second_aux = np.copy(C)
    merge_sort_numba_two(Aux, B, 0, len(B) - 1, second_aux, second)
    return B, second
