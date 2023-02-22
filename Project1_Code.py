import numpy as np


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


def get_errors(A: np.ndarray, U: np.ndarray, S: np.ndarray, Vh: np.ndarray, norm: str) -> np.ndarray:

    size = np.size(S)

    errors = np.zeros(size)

    for k in range(1, size + 1):
        errors[k - 1] = np.linalg.norm(
            A - get_low_rank(U, S, Vh, k), ord=norm) ** 2

    print(errors)
    return errors
