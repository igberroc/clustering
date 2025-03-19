# -*- coding: utf-8 -*-


from points import Distance, euclidean_distance, Point, Cluster
import math
import numpy as np

def min_dist_point_cluster(point: Point, list_clusters: list[Cluster],
                           dist: Distance = euclidean_distance) -> float:
    """
    Given a point and a list of clusters, returns the minimum distance between the point and a cluster,
    where the distance between a point and a cluster is the average distance between the point and all the points.

    Parameters
    ----------
    point: a point.
    list_clusters: list of clusters.
    dist: distance to use.

    Returns
    -------
    minimum distance.
    """
    minimum = math.inf
    b = 0
    for cluster in list_clusters:
        for point_in in cluster.set_points():
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


def silhouette_index(list_clusters: list[Cluster], dist: Distance = euclidean_distance) -> float:
    """
    Given a list of clusters, returns the silhouette index.

    Parameters
    ----------
    list_clusters: list of clusters.
    dist: distance to use.

    Returns
    -------
    silhouette index: between -1 and 1, -1 show bad classification, 1 show good classification.
    """
    num_points = 0
    index = 0
    for _ in range(len(list_clusters)):
        cluster = list_clusters.pop(0)
        for point in cluster.set_points():
            index += silhouette_index_point(point, cluster, list_clusters, dist)
        num_points += cluster.size()
        list_clusters.append(cluster)
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
    centroid = cluster.centroid(dist)
    dispersion = 0
    for point in cluster.set_points():
        dispersion += dist(point,centroid)
    return dispersion / cluster.size(),centroid


def db_index(list_clusters: list[Cluster], dist: Distance = euclidean_distance) -> float:
    """
    Given a list of clusters, returns the db index.

    Parameters
    ----------
    list_clusters: list of clusters.
    dist: distance to use.

    Returns
    -------
    db index: between 0 and infinite, the closer to 0, the better classification.
    """
    k = len(list_clusters)
    list_dispersion = []
    list_centroids = []
    index = 0
    for cluster in list_clusters:
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
                
                
def s_w(list_clusters: list[Cluster], dist: Distance = euclidean_distance) -> tuple[float, int]:
    """
    Given a list of clusters, returns the sum of the distances between all the pairs of points in
    the same cluster, and the number of pairs in the same cluster.

    Parameters
    ----------
    list_clusters: list of clusters.
    dist: distance to use.

    Returns
    -------
    sum of the distances between all the pairs of points in the same cluster, and the number of pairs in the clusters.
    """
    sol = 0
    m = 0
    for cluster in list_clusters:
        m += cluster.size()*(cluster.size()-1)//2
        set_points = cluster.set_points().copy()
        while len(set_points) != 1:
            point = set_points.pop()
            for point_in in set_points:
                sol += dist(point,point_in)

    return sol, m

def s_max_min(data: list[Point], m: int, dist: Distance = euclidean_distance) -> tuple[float,float]:
    """
    Given the set of data and the number of pairs in the clusters (m) , returns the sum of the m minimum distances
    and the sum of the m maximum distances between all the pairs of points in the set of data.

    Parameters
    ----------
    data: set of data.
    m: number of pairs in the clusters.
    dist: distance to use.

    Returns
    -------
    sum of the m minimum distances and the sum of the m maximum distances.
    """

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
    return float(np.sum(minimum_dist)), float(np.sum(maximum_dist))



def c_index(data: list[Point], list_clusters: list[Cluster], dist: Distance = euclidean_distance) -> float:
    """
    Given a set of data and a list of clusters, returns the c-index.

    Parameters
    ----------
    data: set of data.
    list_clusters: list of clusters.
    dist: distance to use.

    Returns
    -------
    c-index: between 0 and 1, the closer to 0, the better classification.
    """
    (s, m) = s_w(list_clusters, dist)
    s_min, s_max = s_max_min(data, m, dist)
    if s_min == s_max:
        return 0
    return (s - s_min) / (s_max - s_min)


def cluster_and_global_centroids(list_clusters: list[Cluster], dist: Distance = euclidean_distance) -> tuple[Point, list[Point]]:
    """
    Given a list of clusters, returns the global centroid of all the points and a list of the centroids
    of each cluster. It is an efficient way to calculate both the global centroid
    and the centroids of each cluster at the same time.

    Parameters
    ----------
    list_clusters: list of clusters.

    Returns
    -------
    The global centroid and a list of the centroids of each cluster.
    """

    if dist == euclidean_distance:
        dimension = list_clusters[0].points_dimension()
        global_centroid = Point.null_point(dimension)
        n = 0
        list_centroids = []
        for cluster in list_clusters:
            points_sum = cluster.points_sum()
            size = cluster.size()
            list_centroids.append(points_sum.div_num(size))
            global_centroid = global_centroid.sum(points_sum)
        global_centroid = global_centroid.div_num(n)
        return global_centroid, list_centroids
    else:
        list_medoids = []
        set_all_points = set()
        for cluster in list_clusters:
            medoid = cluster.centroid(dist)
            list_medoids.append(medoid)
            set_all_points = set_all_points | cluster.set_points()
        all_points_cluster = Cluster(set_all_points)
        global_medoid = all_points_cluster.centroid(dist)
        return global_medoid, list_medoids


def ch_index(list_clusters: list[Cluster], dist: Distance = euclidean_distance) -> float:
    """
    Given a list of clusters, returns the Calinski-Harabasz index.

    Parameters
    ----------
    list_clusters: list of clusters.
    dist: distance to use (euclidean distance recommended).

    Returns
    -------
    ch index: positive number, the higher the better.
    """
    ssb = 0
    ssw = 0
    global_centroid, list_centroids = cluster_and_global_centroids(list_clusters, dist)
    k = len(list_clusters)
    n = 0
    for i in range(k):
        cluster = list_clusters[i]
        n += cluster.size()
        centroid = list_centroids[i]
        ssb += cluster.size()*(dist(centroid,global_centroid)**2)
        for point in cluster.set_points():
            ssw += dist(point, centroid)**2
    return (ssb / (k - 1)) / (ssw / (n - k))


def cluster_distance(cluster1: Cluster, cluster2: Cluster, dist: Distance = euclidean_distance) -> float:
    """
    Given two clusters, returns the distance between them (the smallest distance among the distances
    between points in the two clusters).

    Parameters
    ----------
    cluster1: one cluster.
    cluster2: other cluster.
    dist: distance to use.

    Returns
    -------
    The distance between the two clusters.
    """
    minimum = math.inf
    for point1 in cluster1.set_points():
        for point2 in cluster2.set_points():
            distance = dist(point1, point2)
            if distance < minimum:
                minimum = distance
    return minimum

def diameter(cluster: Cluster, dist: Distance = euclidean_distance) -> float:
    """
    Given a cluster, returns the diameter of the cluster (biggest distance between two points in the cluster).

    Parameters
    ----------
    cluster: the cluster.

    Returns
    -------
    The diameter of the cluster.
    """
    cluster_list = list(cluster.set_points())
    n = len(cluster_list)
    maximum = 0
    for i in range(n):
        point1 = cluster_list[i]
        for j in range(i + 1, n):
            point2 = cluster_list[j]
            distance = dist(point1, point2)
            if distance > maximum:
                maximum = distance
    return maximum

def dunn_index(list_clusters: list[Cluster], dist: Distance = euclidean_distance) -> float:
    """
    Given a list of clusters, returns the Dunn index.

    Parameters
    ----------
    list_clusters: list of clusters.
    dist: distance to use.

    Returns
    -------
    dunn index: positive number, the higher the better.
    """
    k = len(list_clusters)
    max_diameter = 0
    min_distance  = math.inf
    for i in range(k):
        cluster1 = list_clusters[i]
        diam = diameter(cluster1, dist)
        if diam > max_diameter:
            max_diameter = diam
        for j in range(i + 1, k):
            cluster2 = list_clusters[j]
            distance = cluster_distance(cluster1, cluster2, dist)
            if distance < min_distance:
                min_distance = distance
    return min_distance / max_diameter











        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    