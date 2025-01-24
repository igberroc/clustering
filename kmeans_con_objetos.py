# -*- coding: utf-8 -*-


import random
import copy
import math


from points import Distance, Cluster, Distance_euclidea, Point


def diferencia_centroids(centroids, new_centroids, error, dist: Distance = Distance_euclidea) -> bool:
    """
    Given the lists of old and new centroids and a number, returns true if the distance between
    each new and old centroid is less than the given number, and false in other case.

    Parameters
    ----------
    centroids: list of old centroids. 
    new_centroids: list of new centroids. 
    error: number used to compare the distance between centroids.
    dist: distance to use.

    Returns
    -------
    bool: condition satisfied or not.
        
    """
    if centroids == []:
        return True
    for i in range(len(centroids)):
        if dist(centroids[i], new_centroids[i]) > error:
            return True
    return False


def kmeans(data: list[Point], k: int, error: float, max_iterac: int,
           dist: Distance = Distance_euclidea) -> list[Cluster]:
    """
    Given a set of data, number of clusters and conditions for the loop´s body,
    returns the final clusters.

    Parameters
    ----------
    data: list of points.
    k: number of clusters. 
    error: minimum difference between new and old centroids for finishing the loop. 
    max_iterac: maximum number of iterations.
    dist: distance to use.

    Returns
    -------
    lista_clusters: list of final clusters.
    
    Complexity
    -------
    O(N*K*I) where N: number of points
                   K: number of clusters
                   I: number of iterations

    """
    iterac = 0
    lista_clusters = [Cluster() for _ in range(k)]
    centroids = []
    new_centroids = random.sample(data,k)
    while iterac < max_iterac and diferencia_centroids(centroids, new_centroids, error, dist):
        for cluster in lista_clusters:
            cluster.clear()
        centroids = copy.deepcopy(new_centroids)
        new_centroids = []
        for Point in data:
            minimo = math.inf
            i = -1
            for j in range(k):
                d = dist(Point,centroids[j])
                if d < minimo:
                    minimo = d
                    i = j
            cluster = lista_clusters[i]
            cluster.add_Point(Point)
        for cluster in lista_clusters:
            centroid = cluster.centroid()
            new_centroids.append(centroid)
        iterac += 1
    return lista_clusters

