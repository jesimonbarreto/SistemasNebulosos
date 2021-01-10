import numpy as np
import pandas as pd
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from scipy.spatial import distance


# Jesimon, mano, um detalhe importante: segui a orientação de U que o livro segue.
# Ou seja, matrix c x n (c sendo o número de centroides e n sendo o número de features)

# data (NxM) = DataFrame a ser clusterizado N observacoes e M features
def import_data():
    data = loadmat('fcm_dataset.mat')
    data = pd.DataFrame(data['x'])
    lendata = len(data)
    print('Numero de obervacoes: ', lendata)
    nfeatures = data.shape[1]
    print('Numero de atributos (features): ', nfeatures)

    return data.values.transpose()


# n = number of columns, k = number of rows
def generate_u(k: int, n: int):
    return np.random.dirichlet(np.ones(k), size=n)


# returns array of K centroids
def calc_centroids(U, X, m):
    centroids = []

    n_centroids = U.shape[0]
    n_features = U.shape[1]

    for i in range(n_centroids):
        upper_sum = 0
        lower_sum = 0
        for j in range(n_features):
            upper_sum += (U[i][j] ** m) * X[:, j]
            lower_sum += (U[i][j] ** m)
        centroids.append(upper_sum / lower_sum)

    return centroids


def calc_cost(U, X, centroids, m):
    n_centroids = U.shape[0]
    n_features = U.shape[1]

    cost = 0

    for i in range(n_centroids):
        for j in range(n_features):
            dist = distance.euclidean(centroids[i], X[:, j])
            cost += (U[i][j] ** m) * (dist ** 2)

    return cost


def update_u(U, X, centroids, m):
    n_centroids = U.shape[0]
    n_features = U.shape[1]

    for i in range(n_centroids):
        for j in range(n_features):
            dist_kj = 0
            for k in range(n_centroids):
                dist_kj += distance.euclidean(centroids[k], X[:, j])
            dist_ij = distance.euclidean(centroids[i], X[:, j])
            U[i][j] = 1 / (dist_ij / dist_kj ** (2 / (m - 1)))


# x: dataset composto de pontos no plano cartesiano (número de dimensões arbitrário)
# n_centroides: número desejado de clusters
# m: exponente de peso
def fuzzy_k_means(x, n_centroides, m):
    if m <= 1:
        raise Exception(" 'm' should be greater than 1")

    u = generate_u(x.shape[1], n_centroides)
    cost = 1
    centroids = []
    iterations = 0

    while cost > 0.001:
        centroids = calc_centroids(u, x, m)
        cost = calc_cost(u, x, centroids, m)
        update_u(u, x, centroids, m)
        iterations += 1
        print(cost)

    return {'centroids': centroids, 'u_matrix': u, 'n_iterations': iterations}


def main():
    data = import_data()
    result = fuzzy_k_means(data, 5, 2)

    print(result['centroids'])
    print(result['n_iterations'])


if __name__ == "__main__":
    main()
