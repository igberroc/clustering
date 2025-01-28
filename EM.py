# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import multivariate_normal


from puntos import Cluster, Point

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
    u : array with initial means for each distribution (written as arrays).
    covariances : array with covariance matrix for each distribution.
    weights: array with probabilities for belonging to each cluster (each distribution).
    """
    d = len(array[0])
    n = array.size()
    means = array[np.random.choice(n, n_clusters, replace=False)]
    covariances = [np.eye(d) for _ in range(n_clusters)]
    weights = np.ones(n_clusters) / n_clusters
    return means, covariances, weights


def em(data: list[Point], n_clusters: int, eps: float,
       max_iter: int) -> list[Cluster]:
    """
    Given a set of data, the number of clusters, and conditions for the loop´s body,
    returns the list of the clusters.
    
    Parameters
    ----------
    data: list of points
    n_clusters: number of clusters. 
    eps: minimum difference between new log_likelihood and old likelihood for finishing the loop.
    max_iter: maximum number of iterations

    Returns
    -------
    list_clusters: list with final clusters.
    """
    array = np.array([list(point.get_coordinates()) for point in data])
    n = len(data)
    (means, covariances, weights) = initial_parameters(array, n_clusters)
    probability_matrix = np.zeros((n, n_clusters))
    log = 0
    new_log = -np.inf
    iter = 0
    while iter < max_iter and abs(new_log - log) > eps:
        for k in range(n_clusters):
            probability_matrix[:, k] = weights[k] * multivariate_normal.pdf(array, mean = means[k], cov = covariances[k])
        probability_matrix /= probability_matrix.sum(axis = 1, keepdims = True)
        n_k = probability_matrix.sum(axis = 0)
        weights = n_k / n
        for k in range(n_clusters):
            means[k] = (array * probability_matrix[:, k][:, np.newaxis]).sum(axis=0) / n_k[k]
            dif = array - means[k]
            covariances[k] = np.dot((probability_matrix[:, k][:, np.newaxis] * dif).T, dif) / n_k[k]
        log = new_log
        new_log = np.sum(np.log(np.sum([
            weights[k] * multivariate_normal.pdf(array, mean = means[k], cov = covariances[k]) for k in range(n_clusters)
        ], axis=0)))
    list_clusters = [Cluster() for _ in range(n_clusters)]
    sol = np.argmax(probability_matrix, axis = 1)
    for i in range(n):
        Point = data[i]
        j = sol[i]
        list_clusters[j].add_Point(Point)
    return list_clusters
        
        
    
    