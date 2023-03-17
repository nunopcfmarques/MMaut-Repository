import numpy as np
from random import *
import matplotlib.pyplot as plt


def get_low_rank(U: np.ndarray, S: np.ndarray, Vh: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return U_k, S_k, Vh_k


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
        U_k, S_k, Vh_k = get_low_rank(U, S, Vh, k)
        A_err = A - np.linalg.multi_dot([U_k, S_k, Vh_k])
        errors[k - 1] = eval(exp)

    print(str(norm) + ": " + str(errors))
    print(" ")

    return errors


def read_csv(path_str: str) -> list:
    data = []
    lines = []

    with open(path_str) as fp:
        lines = fp.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].split(",")
        lines[i][-1] = lines[i][-1].rstrip()
        data.append(lines[i])

    return data


def data_split(data: np.ndarray, per: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    np.random.shuffle(data)

    size_data = np.size(data, 0)

    test_size = int(round(size_data * per))

    test = data[: test_size]

    train = np.delete(data, np.s_[:test_size], 0)

    return train, test


def movieId_to_i(movieId: int, movieIds: np.ndarray) -> int:
    i = np.where(movieIds == movieId)
    return i


def userId_to_j(userId: int) -> int:
    j = userId - 1
    return j


def getR(train: np.ndarray, userIds: np.ndarray, movieIds: np.ndarray) -> np.ndarray:

    n_rows = np.size(movieIds)
    n_colums = np.size(userIds)

    R = np.zeros((n_rows, n_colums))

    for i in range(np.size(train, 0)):
        userId = int(train[i][0])
        movieId = int(train[i][1])
        rating = float(train[i][2])

        i = movieId_to_i(movieId, movieIds)[0][0]
        j = userId_to_j(userId)

        R[i][j] = rating

    return R


def imput_R(R: np.ndarray, imput_mode: str) -> np.ndarray:

    n_rows = np.size(R, 0)
    n_colums = np.size(R, 1)

    match imput_mode:
        case 'mean_movies':
            for i in range(n_rows):

                ratings = R[i][R[i] > 0]

                if np.size(ratings) == 0:
                    mean = 0

                else:
                    mean = np.mean(ratings)

                R[i][R[i] == 0] = mean

        case 'mean_users':
            for j in range(n_colums):

                ratings = R[:][j][R[:][j] > 0]

                if np.size(ratings) == 0:
                    mean = 0

                else:
                    mean = np.mean(ratings)

                R[:][j][R[:][j] == 0] = mean

    return R


def get_random_arrays(R: np.ndarray, k: int, args: list) -> tuple[np.ndarray, np.ndarray]:

    function = args[0]

    M = eval("np.random." + function +
             "(args[1], args[2], size=(np.size(R, 0), k))")

    U = eval("np.random." + function +
             "(args[1], args[2],  size=(k, np.size(R, 1)))")

    return M, U


def nnls(A, y, epsilon):

    n = np.size(A, 1)
    P = []
    R = list(range(n))
    x = np.zeros(n)
    w = np.dot(A.T, y - np.dot(A, x))

    while len(R) != 0 and max([w[i] for i in R]) > epsilon:

        j = np.where(w == max(w[list(R)]))[0][0]

        P.append(j)
        R.remove(j)

        A_p = A[:, list(P)]

        s = np.zeros(n)

        s[P] = s_p = np.dot(np.linalg.inv(
            np.dot(A_p.T, A_p)), np.dot(A_p.T, y))

        s[R] = np.zeros(np.size(R))

        while min(s_p) <= 0:
            alpha = min((x[i] - s_p[i])/(1 - s_p[i]) for i in P if s_p[i] <= 0)
            x = x + alpha * (s - x)

            move_j = [j for j in P if x[j] <= 0]
            for j in move_j:
                P.remove(j)
                R.append(j)

            A_p = A[:, P]

            s[P] = s_p = np.dot(np.linalg.inv(
                np.dot(A_p.T, A_p)), np.dot(A_p.T, y))

            s[R] = np.zeros(np.size(R))

        x = s
        w = np.dot(A.T, y - np.dot(A, x))

    return x


def sgd(M: np.ndarray, U: np.ndarray, R: np.ndarray, iters: int = 10, gamma: float = 0.001, lmbda: float = 0.01) -> tuple[np.ndarray, np.ndarray]:

    movies, users = np.nonzero(R)

    for iter in range(iters):

        for i, j in zip(movies, users):

            err = R[i][j] - (np.dot(M[i], U[:, j]))
            dMi = M[i]
            M[i] = (M[i] + gamma *
                    (err * U[:, j].T - lmbda * M[i]))
            U[:, j] = U[:, j] + gamma * \
                (err * dMi.T - lmbda * U[:, j])

    return M, U


def als(M: np.ndarray, U: np.ndarray, R: np.ndarray, iters: int = 10, lmbda: float = 0.01) -> tuple[np.ndarray, np.ndarray]:

    k = np.size(U, 0)

    for iter in range(iters):

        for i in range(np.size(R, 0)):
            users = np.nonzero(R[i])[0]
            sum_y = np.zeros((k, k))
            sum_r_y = np.zeros((k, ))

            for u in users:
                sum_y += np.outer(U[:, u], U[:, u])
                sum_r_y += R[i][u] * U[:, u]

            M[i] = nnls(sum_y + lmbda * np.eye(k), sum_r_y, 0.00001)
            M[M < 0] = 0

        for j in range(np.size(R, 1)):
            movies = np.nonzero(R[:, j])[0]
            sum_x = np.zeros((k, k))
            sum_r_x = np.zeros((k, ))

            for m in movies:
                sum_x += np.outer(M[m], M[m])
                sum_r_x += R[m][j] * M[m]

            U[:, j] = nnls(sum_x + lmbda * np.eye(k), sum_r_x, 0.00001)

    return M, U


def get_metrics(train: np.ndarray, test: np.ndarray, M: np.ndarray, U: np.ndarray, movieIds: np.ndarray) -> tuple[float, float, float, float]:

    RMSE_list_train = []
    MAE_list_train = []

    RMSE_list_test = []
    MAE_list_test = []

    R = np.dot(M, U)

    for i in range(np.size(train, 0)):
        userId = int(train[i][0])
        movieId = int(train[i][1])

        m = movieId_to_i(movieId, movieIds)[0][0]
        u = userId_to_j(userId)

        rating = float(train[i][2])

        err = rating - R[m][u]

        RMSE_list_train.append((err) ** 2)
        MAE_list_train.append(np.abs(err))

    RMSE_train = np.round(np.sqrt(np.mean(RMSE_list_train)), 5)
    MAE_train = np.round(np.mean(MAE_list_train), 5)

    for j in range(np.size(test, 0)):
        userId = int(test[j][0])
        movieId = int(test[j][1])

        m = movieId_to_i(movieId, movieIds)[0][0]
        u = userId_to_j(userId)

        rating = float(test[j][2])

        err = rating - R[m][u]

        RMSE_list_test.append((err) ** 2)
        MAE_list_test.append(np.abs(err))

    RMSE_test = np.round(np.sqrt(np.mean(RMSE_list_test)), 5)
    MAE_test = np.round(np.mean(MAE_list_test), 5)

    print("Train error metrics: " + "RMSE: " +
          str(RMSE_train) + "; " "MAE: " + str(MAE_train) + ".")
    print("Test error metrics: " + "RMSE: " +
          str(RMSE_test) + "; " "MAE: " + str(MAE_test) + ".")

    return RMSE_train, RMSE_test, MAE_train, MAE_test


def plot_movies(M: np.ndarray, movies: list, movieIds: np.ndarray):

    ids = [int(movie_ids[0]) for movie_ids in movies]
    names = [movie_ids[1] for movie_ids in movies]

    k1s = []
    k2s = []

    for id in ids:

        i = movieId_to_i(id, movieIds)[0][0]

        k1s.append(M[i][0])
        k2s.append(M[i][1])

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    ax.scatter(k1s, k2s)

    for i, name in enumerate(names):
        ax.annotate(name, (k1s[i], k2s[i]))

    plt.plot()
