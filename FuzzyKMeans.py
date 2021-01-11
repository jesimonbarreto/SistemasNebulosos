import numpy as np
import pandas as pd
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from scipy.spatial import distance
import copy
import matplotlib.pyplot as plt


#Variavel para controle de plot dos resuldos e prints durante o código
debug = False

# data (NxM) = DataFrame a ser clusterizado N observacoes e M features
def import_data():
    data = loadmat('fcm_dataset.mat')
    data = pd.DataFrame(data['x'])
    lendata = len(data)
    print('Numero de obervacoes: ', lendata)
    nfeatures = data.shape[1]
    print('Numero de atributos (features): ', nfeatures)

    return data.values#.transpose()

def reduc_samples(X,n_samples=10):
    return X[:n_samples, :]

def plot_samples(x, u, centroids):
    y_kmeans = np.argmax(u, axis=1)
    plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

# n = number of columns, k = number of rows
def generate_u(k: int, n: int):
    return np.random.dirichlet(np.ones(k), size=n)


# returns array of K centroids
def calc_centroids(U, X, m):
    centroids = []
    n_centroids = U.shape[1]
    n_samples = U.shape[0]

    for i in range(n_centroids):
        u_i = U[:,i] 
        ui_m = u_i ** m

        features_cent = []
        n_feat_cent = int(X.shape[1])
        for feat in range(n_feat_cent):
            x_feat = X[:,feat]
            feat_cent = np.sum((ui_m*x_feat))/np.sum(ui_m) 
            features_cent.append(feat_cent)
        centroids.append(features_cent)

    centroids = np.array(centroids)
    return centroids


def calc_cost(U, X, centroids, m):
    n_centroids = U.shape[1]
    n_samples = U.shape[0]

    cost = 0

    for i in range(n_centroids):
        for j in range(n_samples):
            dist = distance.euclidean(centroids[i], X[j,:])
            cost += (U[j][i] ** m) * (dist ** 2)

    return cost


def update_u(U, X, centroids, m):
    n_centroids = U.shape[1]
    n_samples = U.shape[0]

    U_new = copy.copy(U)

    for j in range(n_samples):
        dist_kj = 0
        for k in range(n_centroids):
            dist_kj += distance.euclidean(centroids[k], X[j,:])
        for i in range(n_centroids):
            dist_ij = distance.euclidean(centroids[i], X[j,:])
            U_new[j][i] = 1 / (dist_ij / dist_kj) ** (2/(m - 1))
        #normaliza valores entre 0 e 1 para cada amostra
        sum_ = np.sum(U_new[j])
        U_new[j] = U_new[j]/sum_
    
    return U_new

# x: dataset composto de pontos no plano cartesiano (número de dimensões arbitrário)
# n_centroides: número desejado de clusters
# m: exponente de peso
def fuzzy_k_means(x, n_centroides, m, threshold = 0.001, max_inter = 10):
    if m <= 1:
        raise Exception(" 'm' should be greater than 1")

    u = generate_u(n_centroides, x.shape[0])
    cost = 1
    centroids = []
    iterations = 0

    while cost > threshold and iterations < max_inter:
        centroids = calc_centroids(u, x, m)
        cost = calc_cost(u, x, centroids, m)
        u = update_u(u, x, centroids, m)
        iterations += 1
        print(cost)
        if debug:
            print('Matrix centroids: {}'.format(centroids))
            print('Matrix U: {}'.format(u))
            plot_samples(x, u, centroids)
        
    
    return {'centroids': centroids, 'u_matrix': u, 'n_iterations': iterations}


def main():
    data = import_data()
    if debug:
        data = reduc_samples(data, n_samples=10)
    result = fuzzy_k_means(data, 4, 2, 0.0001, 20)

    print(result['centroids'])
    print(result['n_iterations'])


if __name__ == "__main__":
    main()
