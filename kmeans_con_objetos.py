# -*- coding: utf-8 -*-


import random
import copy
import math

from puntos import Distancia, Cluster, distancia_euclidea, Punto

def diferencia_centroides(centroides, new_centroides, error, dist: Distancia = distancia_euclidea) -> bool:
    """
    Given the lists of old and new centroids and a number, returns true if the distance between
    each new and old centroid is less than the given number, and false in other case.

    Parameters
    ----------
    centroides: list of old centroids. 
    new_centroides: list of new centroids. 
    error: number used to compare the distance between centroids.
    dist: distance to use.

    Returns
    -------
    bool: condition satisfied or not.
        
    """
    if centroides == []:
        return True
    for i in range(len(centroides)):
        if dist(centroides[i], new_centroides[i]) > error:
            return True
    return False


def kmeans(datos: list[Punto], k: int, error: float, max_iterac: int, 
           dist: Distancia = distancia_euclidea) -> list[Cluster]:
    """
    Given a set of data, number of clusters and conditions for the loop´s body,
    returns the final clusters.

    Parameters
    ----------
    datos: list of points.
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
    centroides = []
    new_centroides = random.sample(datos,k)
    while iterac < max_iterac and diferencia_centroides(centroides, new_centroides, error, dist):
        for cluster in lista_clusters:
            cluster.vaciar()
        centroides = copy.deepcopy(new_centroides)
        new_centroides = []
        for punto in datos:
            minimo = math.inf
            i = -1
            for j in range(k):
                d = dist(punto,centroides[j])
                if d < minimo:
                    minimo = d
                    i = j
            cluster = lista_clusters[i]
            cluster.agregar_punto(punto)
        for cluster in lista_clusters:
            centroide = cluster.calcular_centroide()
            new_centroides.append(centroide)
        iterac += 1
    return lista_clusters

