import numpy as np
from numba import njit


def is_diagonally_dominant(A):
    d = np.abs(np.diag(A))
    s = np.sum(np.abs(A), axis=1) - d

    return np.all(d > s)

@njit
def gaussian(A,b):

    n = len(A)
    for k in range(n-1):    #pivot
        if A[k,k] == 0:
            raise ValueError("Zero pivot encountered")
        d = 1/A[k,k]
        for i in range(k+1,n):  #every row below pivot
            A[i,k] *= d        #use the values below pivot to store l
            b[i] -= b[k]*A[i,k]
            for j in range(k+1,n):  #every column to left of pivot
                A[i,j] -= A[i,k]*A[k,j]     #a[i,j] -= l[i,k]*a[k,j]

    if A[n-1,n-1] == 0:
        raise ValueError("Zero pivot encountered")
    return A,b


@njit
 
def gaussian_with_partial_pivoting(A,b):
    n = len(A)

    for k in range(n-1):

        r = k + np.argmax(np.abs(A[k:,k]))

        if r != k:
            for j in range(n):
                A[k,j], A[r,j] = A[r,j], A[k,j]

            b[k], b[r] = b[r], b[k]

        if A[k,k] == 0:
            print("Zero pivot:", k, r)
            raise ValueError("Zero pivot encountered")

        d = 1/A[k,k]

        for i in range(k+1,n):
            A[i,k] *= d
            b[i] -= b[k]*A[i,k]

            for j in range(k+1,n):
                A[i,j] -= A[i,k]*A[k,j]

    if A[n-1,n-1] == 0:
        raise ValueError("Zero pivot encountered")
    return A,b