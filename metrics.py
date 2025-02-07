# -*- coding: utf-8 -*-


from points import Distance, euclidean_distance, Point, Cluster
import math
import copy
import numpy as np

def min_dist_point_cluster(point: Point, list_cluster: list[Cluster],
                              dist: Distance = euclidean_distance) -> float:
    """
    Given a point and a list of clusters, returns the minimum distance between the point and a clusters,
    where the distance between a point and a cluster is the average distance between the point and all the points.

    Parameters
    ----------
    point: a point.
    list_cluster: list of clusters.
    dist: distance to use.

    Returns
    -------
    minimum distance.
    """
    minimum = math.inf
    b = 0
    for cluster in list_cluster:
        for point_in in cluster.points:
            b += dist(point,point_in)
        b = b / cluster.size()
        minimum = min(b,minimum)
        b = 0
    return minimum


def avg_dist_point_cluster(point: Point, cluster: Cluster, dist: Distance = euclidean_distance) -> float:
    """
    Given a point and the cluster where the point belongs to, returns the average distance between the point and all the points in the cluster.

    Parameters
    ----------
    point: a point.
    cluster: cluster where the point belongs to.
    dist: distance to use.

    Returns
    -------
    average distance.
    """
    a = 0
    for point_in in cluster.points:
        a += dist(point,point_in)
    if cluster.size() != 1:
        a = a / (cluster.size()-1)
    return a
            


def silhouette_index_point(point: Point, own_cluster: Cluster,
                         rest_clusters: list[Cluster], dist: Distance = euclidean_distance ) -> float:
    """
    Given a point, the cluster where the point belongs to and the rest of clusters, returns the silhouette index of the point.

    Parameters
    ----------
    point: a point.
    own_cluster: cluster where the point belongs to.
    rest_clusters: rest of clusters.
    dist: distance to use.

    Returns
    -------
    silhouette index of a point.
    """
    b = min_dist_point_cluster(point,rest_clusters, dist)
    a = avg_dist_point_cluster(point, own_cluster, dist)
    index = (b - a) / max(a,b)
    return index

# -1 indica mala clasficacion, 1 indica buena clasificacion
def silhouette_index(list_cluster: list[Cluster], dist: Distance = euclidean_distance) -> float:
    """
    Given a list of clusters, returns the silhouette index.

    Parameters
    ----------
    list_cluster: list of clusters.
    dist: distance to use.

    Returns
    -------
    silhouette index: between -1 and 1, -1 show bad classification, 1 show good classification.
    """
    num_points = 0
    index = 0
    for _ in range(len(list_cluster)):
        cluster = list_cluster.pop(0)
        for point in cluster.points:
            index += silhouette_index_point(point, cluster, list_cluster, dist)
        num_points += cluster.size()
        list_cluster.append(cluster)
    return index / num_points
        
    
def cluster_dispersion(cluster: Cluster, dist: Distance = euclidean_distance) -> tuple[float, Point]:
    """
    Given a cluster, returns the dispersion and the centroid.
    Parameters
    ----------
    cluster: a cluster.
    dist: distance to use.

    Returns
    -------
    dispersion and centroid.
    """
    centroid = cluster.centroid()
    dispersion = 0
    for point in cluster.points:
        dispersion += dist(point,centroid)
    return dispersion / cluster.size(),centroid


def db_index(list_cluster: list[Cluster], dist: Distance = euclidean_distance) -> float:
    """
    Given a list of clusters, returns the db index.

    Parameters
    ----------
    list_cluster: list of clusters.
    dist: distance to use.

    Returns
    -------
    db index: between 0 and infinite, the closer to 0, the better classification.
    """
    k = len(list_cluster)
    list_dispersion = []
    list_centroids = []
    index = 0
    for cluster in list_cluster:
        (dispersion, centroid) = cluster_dispersion(cluster,dist)
        list_dispersion.append(dispersion)
        list_centroids.append(centroid)
    for i in range(k):
        maximum = 0
        for j in range(k):
            if i != j:
                formula = (list_dispersion[i] + list_dispersion[j]) / dist(list_centroids[i],list_centroids[j])
                maximum = max(maximum, formula)
        index += maximum
    return index / k
                
                
def s_w(list_cluster: list[Cluster], dist: Distance = euclidean_distance) -> tuple[float, int]:
    sol = 0
    m = 0
    for cluster in list_cluster:
        m += cluster.size()*(cluster.size()-1)//2
        set_points = cluster.points.copy()
        while len(set_points) != 1:
            point = set_points.pop()
            for point_in in set_points:
                sol += dist(point,point_in)

    return (sol, m)

def s_max_min(data: list[Point], m: int, dist: Distance = euclidean_distance) -> tuple[float,float]:
    n_pairs = len(data)*(len(data)-1)//2
    array_dist = np.zeros(n_pairs)
    k = 0
    for i in range(len(data)):
        point = data[i]
        for j in range(i+1,len(data)):
            point2 = data[j]
            array_dist[k] = dist(point,point2)
            k += 1
    minimum_dist = np.partition(array_dist, m)[:m]
    maximum_dist = np.partition(array_dist, -m)[-m:]
    return np.sum(minimum_dist), np.sum(maximum_dist)



def c_index(data: list[Point], list_cluster: list[Cluster], dist: Distance = euclidean_distance) -> float:
    (s, m) = s_w(list_cluster, dist)
    s_min, s_max = s_max_min(data, m, dist)
    return (s - s_min) / (s_max - s_min)









        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    