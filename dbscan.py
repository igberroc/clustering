# -*- coding: utf-8 -*-

from puntos import Punto, distancia_euclidea,Distancia,Cluster

def vecinos(datos: list[Punto], punto: Punto, eps: float,
            dist: Distancia = distancia_euclidea) -> list[Punto]:
    """
    Given a set of data, a point and epsilon, returns the list of neighbors.
    Parameters
    ----------
    datos : list with data
    punto: point to search neighbors.
    eps: epsilon value for dbscan
    dist : distance to use

    Returns
    -------
    vecinos: list of neighbors

    """
    vecinos = []
    for p in datos:
        if dist(p,punto) <= eps and p != punto:
            vecinos.append(p)
    return vecinos
            
    
def dbscan(datos: list[Punto], eps: float, minPts: int, 
           dist: Distancia = distancia_euclidea) -> tuple[list[Cluster], Cluster]:
    """
    Given a set of data, epsilon and minimum number of points to form a neighborhood,
    returns the list of clusters and the noise cluster.
    Parameters
    ----------
    datos : list with data
    eps: epsilon value for dbscan
    minPts: minimum number of points to form a neighborhood,
    dist : distance to use

    Returns
    -------
    (lista_clusters, ruido) where
    lista_clusters: list of clusters
    ruido: cluster with noise points.

    """
    visitados = {punto: False for punto in datos}
    lista_clusters = []
    ruido = Cluster()
    for punto in datos:
        if not(visitados[punto]):
            visitados[punto] = True
            v = vecinos(datos, punto, eps, dist)
            if len(v) + 1 >= minPts:
                cluster = Cluster({punto})                
                while len(v) > 0:
                    vecino = v.pop()
                    if not(visitados[vecino]):
                        visitados[vecino] = True
                        if vecino in ruido.puntos:
                            ruido.quitar_punto(vecino)
                        cluster.agregar_punto(vecino)
                        new_v = vecinos(datos, vecino, eps, dist)
                        if len(new_v) + 1 >= minPts:
                            v = v + new_v
                lista_clusters.append(cluster)
            else:
                ruido.agregar_punto(punto)
    return (lista_clusters, ruido)
            
                        
                     
                    
                    
                    
                