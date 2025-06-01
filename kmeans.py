# -*- coding: utf-8 -*-

import random
import copy
import math


from points import Distance, Cluster, euclidean_distance, Point


def centroids_condition(centroids: list[Point], new_centroids: list[Point], eps: float, dist: Distance = euclidean_distance) -> bool:
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


def kmeans(data: list[Point], k: int, eps: float, max_iter: int,
           dist: Distance = euclidean_distance) -> list[Cluster]:
    """
    Given a set of data, the number of clusters and conditions for the loop´s body,
    returns the final clusters.

    Parameters
    ----------
    data: list of points.
    k: number of clusters. 
    eps: minimum difference between new and old centroids for finishing the loop. 
    max_iter: maximum number of iterations.
    dist: distance to use.

    Returns
    -------
    list of final clusters.
    
    Complexity
    -------
    O(N*K*d*I) where N: number of points.
                   K: number of clusters.
                   d: dimensionality of data.
                   I: maximum number of iterations.

    """
    iter = 0
    list_clusters = [Cluster() for _ in range(k)]
    centroids = []
    new_centroids = random.sample(data,k)
    while iter < max_iter and centroids_condition(centroids, new_centroids, eps, dist):
        for cluster in list_clusters:
            cluster.clear()
        centroids = copy.deepcopy(new_centroids)
        new_centroids = [0 for _ in range(k)]
        for point in data:
            minimum = math.inf
            i = -1
            for j in range(k):
                d = dist(point,centroids[j])
                if d < minimum:
                    minimum = d
                    i = j
            cluster = list_clusters[i]
            cluster.add_point(point)
        for i in range(k):
            cluster = list_clusters[i]
            centroid = cluster.centroid(dist)
            new_centroids[i] = centroid
        iter += 1
    return list_clusters

