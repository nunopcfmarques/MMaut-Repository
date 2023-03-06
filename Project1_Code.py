import numpy as np
from random import *


def get_low_rank(U: np.ndarray, S: np.ndarray, Vh: np.ndarray, k: int) -> np.ndarray:
    '''
    Input:
    U, S, Vh: np.ndarray's resulting from SVD
    k: An integer for the reduce rank

    Output:
    A_k: Reduced np.ndarray 
    '''

    # Choose first k colums of U
    U_k = U[:, :k]

    # Choose first k rows of Vh
    Vh_k = Vh[:k, :]

    # Choose the subdiagonal k x k in S
    S_k = np.diag(S)[:k, :k]

    # Make matrix

    A_k = np.linalg.multi_dot([U_k, S_k, Vh_k])

    return A_k


def get_errors(A: np.ndarray, U: np.ndarray, S: np.ndarray, Vh: np.ndarray, norm: str) -> np.ndarray:
    '''
    Input: 
    U, S, Vh: np.ndarray's resulting from SVD
    norm: A string for the norm

    Output:
    errors: np.ndarray with the errors from 1 to size(S)
    '''
    size = np.size(S)

    errors = np.zeros((size))

    if norm == 'sqrt':
        exp = "np.sum(np.square(A_err))"

    else:
        exp = "np.linalg.norm(A_err, ord=norm)"

    for k in range(1, size + 1):
        A_err = A - get_low_rank(U, S, Vh, k)
        errors[k - 1] = eval(exp)

    print(str(norm) + ": " + str(errors))
    print(" ")

    return errors
