import numpy as np
from random import *


def get_low_rank(U: np.ndarray, S: np.ndarray, Vh: np.ndarray, k: int) -> np.ndarray:

    # Choose first k colums of U
    U_k = U[:, :k]

    # Choose first k rows of Vh
    Vh_k = Vh[:k, :]

    # Choose the subdiagonal k x k in S
    S_k = np.diag(S)[:k, :k]

    # Make matrix
    A_k = (np.dot(U_k, S_k))

    A_k = (np.dot(A_k, Vh_k))

    return A_k


def get_errors(A: np.ndarray, U: np.ndarray, S: np.ndarray, Vh: np.ndarray, norm: str) -> np.ndarray[np.ndarray, np.ndarray]:

    size = np.size(S)

    # erros in array with two sub-arrays each of size k.
    # the first list is the norm of the error
    # the second is the sum of squared residuals
    errors = np.zeros((2, size))

    for k in range(1, size + 1):
        A_err = A - get_low_rank(U, S, Vh, k)
        errors[0][k - 1] = np.linalg.norm(
            A_err, ord=norm)
        errors[1][k - 1] = np.sum(np.square(A_err))

    return errors
