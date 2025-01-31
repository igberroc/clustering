# -*- coding: utf-8 -*-

import numpy as np
import copy

from points import Distance, euclidean_distance, Point


def centroids_condition(centroids: list[Point], new_centroids: list[Point],
                        eps: float, dist: Distance = euclidean_distance) -> bool:
    """
    Given the lists of old and new centroids and a number, returns true if the distance between
    each new and old centroid is less than the given number, and false in other case.

    Parameters
    ----------
    centroids: list of old centroids.
    new_centroids: list of new centroids.
    eps: number used to compare the distance between centroids.
    dist: distance to use.
    Returns
    -------
    condition satisfied or not.

    """
    if centroids == []:
        return True
    for i in range(len(centroids)):
        if dist(centroids[i], new_centroids[i]) > eps:
            return True
    return False


def fuzzy_cmeans(data: list[Point], initial_centroids: list[Point], m: int, c: int, eps: float, max_iter: int,
                 dist: Distance = euclidean_distance) -> np.ndarray:
    """
    Given a set of data, a list of initial centroids, the fuzzification parameter,
    the number of clusters and conditions for the loop´s body, returns the fuzzy partition matrix.

    Parameters
    ----------
    data : list of points.
    initial_centroids : list of initial centroids (needed because if any centroid is the same as one of the data points,
                                                   the algorithm will fail).
    m : fuzzification parameter (usually set to 2).
    c : number of clusters.
    eps : minimum difference between new and old centroids for finishing the loop.
    max_iter : maximum number of iterations.
    dist : distance to use.

    Returns
    -------
    fuzzy partition matrix.

    Complexity
    -------
    O(N*(C^2)*I) where N: number of points.
                   C: number of clusters.
                   I: maximum number of iterations.

    """
    n = len(data)
    iter = 0
    centroids = []
    new_centroids = initial_centroids
    membership_matrix = np.zeros((c,n))
    while iter < max_iter and centroids_condition(centroids, new_centroids, eps, dist):
        centroids = copy.deepcopy(new_centroids)
        new_centroids = [0 for _ in range(c)]
        for j in range(n):
            for i in range(c):
                s = 0
                for l in range(c):
                    coeff = dist(data[j], centroids[l]) / dist(data[j],centroids[i])
                    coeff = coeff ** (1/(1 - m))
                    s += coeff
                membership_matrix[i,j] = 1 / s
        for i in range(c):
            div = 0
            for j in range(n):
                point = data[j]
                u = membership_matrix[i,j] ** m
                if j == 0:
                    points_sum = point.mul_num(u)
                else:
                    points_sum = points_sum.sum(point.mul_num(u))
                    div += u
            new_centroids[i] = points_sum.div_num(div)
        iter += 1
    return membership_matrix
        
        
                
            
        
        
                
                
                
            
            
        
    