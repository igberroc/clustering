# -*- coding: utf-8 -*-

from points import Point, euclidean_distance,Distance,Cluster

def vecinos(data: list[Point], Point: Point, eps: float,
            dist: Distance = euclidean_distance) -> list[Point]:
    """
    Given a set of data, a point and epsilon, returns the list of neighbors.
    Parameters
    ----------
    data : list with data
    Point: point to search neighbors.
    eps: epsilon value for dbscan
    dist : distance to use

    Returns
    -------
    vecinos: list of neighbors

    """
    vecinos = []
    for p in data:
        if dist(p,Point) <= eps and p != Point:
            vecinos.append(p)
    return vecinos
            
    
def dbscan(data: list[Point], eps: float, minPts: int, 
           dist: Distance = euclidean_distance) -> tuple[list[Cluster], Cluster]:
    """
    Given a set of data, epsilon and minimum number of points to form a neighborhood,
    returns the list of clusters and the noise cluster.
    Parameters
    ----------
    data : list with data
    eps: epsilon value for dbscan
    minPts: minimum number of points to form a neighborhood,
    dist : distance to use

    Returns
    -------
    (list_clusters, ruido) where
    list_clusters: list of clusters
    ruido: cluster with noise points.

    """
    visitados = {Point: False for Point in data}
    list_clusters = []
    ruido = Cluster()
    for Point in data:
        if not(visitados[Point]):
            visitados[Point] = True
            v = vecinos(data, Point, eps, dist)
            if len(v) + 1 >= minPts:
                cluster = Cluster({Point})                
                while len(v) > 0:
                    vecino = v.pop()
                    if not(visitados[vecino]):
                        visitados[vecino] = True
                        if vecino in ruido.points:
                            ruido.quit_Point(vecino)
                        cluster.add_Point(vecino)
                        new_v = vecinos(data, vecino, eps, dist)
                        if len(new_v) + 1 >= minPts:
                            v = v + new_v
                list_clusters.append(cluster)
            else:
                ruido.add_Point(Point)
    return (list_clusters, ruido)
            
                        
                     
                    
                    
                    
                