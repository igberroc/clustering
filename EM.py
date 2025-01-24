# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import multivariate_normal


from points import Cluster, Point

def inicializacion(array: np.ndarray, n_clusters: int,
                   n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given the array with points, the number of clusters and the number of points,
    returns initial parameters for the gaussian distributions.
    
    Parameters
    ----------
    array : array with points(written as arrays)
    n_clusters : number of clusters
    n : number of points
    
    Returns
    -------
    u : array with initial means for each distribution (written as arrays)
    covarianzas : array with covariance matrix for each distribution
    pesos: array with probabilities for belonging to each cluster (each distribution)
    """
    d = len(array[0])
    indices = np.random.choice(n, n_clusters, replace=False)
    u = array[indices]
    covarianzas = [np.eye(d) for _ in range(n_clusters)]
    pesos = np.ones(n_clusters) / n_clusters
    return (u,covarianzas,pesos)


def em(datos: list[Point], n_clusters: int, error: float,
       max_iterac: int) -> list[Cluster]:
    """
    Given a set of data, the number of clusters, and conditions for the loop´s body,
    returns the list of the clusters.
    
    Parameters
    ----------
    datos: list of points
    n_clusters: number of clusters. 
    error: minimum difference between new log_likelihood and old likelihood for finishing the loop.
    max_iterac: maximum number of iterations

    Returns
    -------
    lista_clusters: list with final clusters.
    """
    array = np.array([list(Point.get_coordinates()) for Point in datos])
    n = len(datos)
    (u,covarianzas,pesos) = inicializacion(array, n_clusters,n)
    matriz_prob = np.zeros((n, n_clusters))
    log = 0
    new_log = -np.inf
    iterac = 0
    while iterac < max_iterac and abs(new_log - log) > error:
        for k in range(n_clusters):
            matriz_prob[:, k] = pesos[k] * multivariate_normal.pdf(array, mean = u[k], cov = covarianzas[k])
        matriz_prob /= matriz_prob.sum(axis = 1, keepdims = True)
        n_k = matriz_prob.sum(axis = 0)
        pesos = n_k / n
        for k in range(n_clusters):
            u[k] = (array * matriz_prob[:, k][:, np.newaxis]).sum(axis=0) / n_k[k]  
            dif = array - u[k]
            covarianzas[k] = np.dot((matriz_prob[:, k][:, np.newaxis] * dif).T, dif) / n_k[k]
        log = new_log
        new_log = np.sum(np.log(np.sum([
            pesos[k] * multivariate_normal.pdf(array, mean = u[k], cov = covarianzas[k]) for k in range(n_clusters)
        ], axis=0)))
    lista_clusters = [Cluster() for _ in range(n_clusters)]
    sol = np.argmax(matriz_prob, axis = 1)
    for i in range(n):
        Point = datos[i]
        j = sol[i]
        lista_clusters[j].add_Point(Point)
    return lista_clusters
        
        
    
    