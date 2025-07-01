# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import multivariate_normal


from points import Cluster, Point

def initial_parameters(array: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given an array with points, the number of clusters and the number of points,
    returns initial parameters for the gaussian distributions.
    
    Parameters
    ----------
    array : array with points (written as arrays).
    n_clusters : number of clusters.
    
    Returns
    -------
    means : array with initial means for each distribution (written as arrays).
    covariances : array with covariance matrix for each distribution.
    weights: array with probabilities for belonging to each cluster (each distribution).
    """
    d = len(array[0])
    n = len(array)
    means = array[np.random.choice(n, n_clusters, replace=False)]
    covariances = [np.eye(d) for _ in range(n_clusters)]
    weights = np.ones(n_clusters) / n_clusters
    return means, covariances, weights


def em(data: list[Point], n_clusters: int, initial_covariances: list[np.ndarray],
       eps: float, max_iter: int) -> list[Cluster]:
    """
    Given a set of data, the number of clusters, and conditions for the loop´s body,
    returns the list of the clusters using the Expectation-Maximization algorithm.
    
    Parameters
    ----------
    data: list of points.
    n_clusters: number of clusters.
    initial_covariances: list of initial covariance matrices.
    eps: minimum difference between new log_likelihood and old likelihood for finishing the loop.
    max_iter: maximum number of iterations.

    Returns
    -------
    list_clusters: list with final clusters.

    Complexity
    -------
    O(I*N*K*d^2) where I: number of iterations.
                   N: number of points.
                   K: number of clusters.
                   d: dimension of the points.

    """
    array = np.array([list(point.get_coordinates()) for point in data])
    n = len(data)
    weights = np.ones(n_clusters) / n_clusters
    covariances = initial_covariances
    means = array[np.random.choice(n, n_clusters, replace=False)]
    probability_matrix = np.zeros((n, n_clusters))
    log = 0
    new_log = -np.inf
    iter = 0
    while iter < max_iter and abs(new_log - log) > eps:
        for k in range(n_clusters):
            probability_matrix[:, k] = weights[k] * multivariate_normal.pdf(array, mean = means[k], cov = covariances[k])
        probability_matrix /= probability_matrix.sum(axis = 1, keepdims = True)  #Posterior probability (E step).
        n_k = probability_matrix.sum(axis = 0)
        weights = n_k / n       #New weights
        for k in range(n_clusters): # and new parameters (M step).
            means[k] = (array * probability_matrix[:, k][:, np.newaxis]).sum(axis=0) / n_k[k]       #New means.
            dif = array - means[k]
            covariances[k] = np.dot((probability_matrix[:, k][:, np.newaxis] * dif).T, dif) / n_k[k]    #New covariances.
        log = new_log
        new_log = np.sum(np.log(np.sum([
            weights[k] * multivariate_normal.pdf(array, mean = means[k], cov = covariances[k]) for k in range(n_clusters)
        ], axis=0)))
        iter += 1
    list_clusters = [Cluster() for _ in range(n_clusters)]
    sol = np.argmax(probability_matrix, axis = 1)    # Index of cluster with maximum probability for each point.
    for i in range(n):
        point = data[i]
        j = sol[i]
        list_clusters[j].add_point(point)
    return list_clusters
        
        
    
    