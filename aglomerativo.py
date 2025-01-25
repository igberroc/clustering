# -*- coding: utf-8 -*-



import math
import numpy as np

from puntos import Distance, Cluster, euclidean_distance, Point

def proximity_matrix(data:list[Point], dist: Distance = euclidean_distance) -> tuple[np.ndarray,float,tuple[int,int]]:
    """
    Given a set of data and the distance to use, returns the initial proximity matrix, the minimum value
    of the matrix and the coordinates of the minimum value.
    
    Parameters
    ----------
    data: set of data.
    dist: distance to use.

    Returns
    -------
    proximity matrix, minimum value and coordinates of minimum value.
    """
    n = len(data)
    matrix = np.zeros((n,n))
    minimum = math.inf
    i_minimum = 0
    j_minimum = 1
    for i in range(n-1):
        for j in range(i+1,n):
            d = dist(data[i],data[j])
            matrix[i,j] = d
            if d < minimum:
                i_minimum = i
                j_minimum = j
                minimum = d
    return matrix,minimum,(i_minimum,j_minimum)
    
def min_distance(matrix: np.ndarray) -> tuple[float,tuple[int,int]]:
    """
    Given the proximity matrix, returns the minimum value and the coordinates of the minimum value.
    
    Parameters
    ----------
    matrix: proximity matrix.

    Returns
    -------
    minimum value and coordinates of minimum value.
    """
    n = len(matrix)
    minimum = math.inf
    i_minimum = 0
    j_minimum = 1
    for i in range(n-1):
        for j in range(i+1,n):
            if matrix[i,j] < minimum:
                i_minimum = i
                j_minimum = j
                minimum = matrix[i,j]
    return minimum,(i_minimum,j_minimum)
            
def lance_williams(matrix:np.ndarray, l:int, i:int, j:int, a:float, b:float, c:float, d:float) -> float:
    """
    Given the proximity matrix, the index of the cluster for which we want to calculate the distance, the indexes
    of both clusters that are going to be combined, and the parameters for lance williams formula, returns the distance
    between one cluster and the combination of the other two.
    
    Parameters
    ----------
    matrix: proximity matrix.
    l: index of the cluster for which we want to calculate the distance.
    i: index of one of the clusters that is going to be combined.
    j: index of the other cluster that is going to be combined.
    a: parameter for lance williams.
    b: parameter for lance williams.
    c: parameter for lance williams.
    d: parameter for lance williams.

    Returns
    -------
    distance between one cluster and the combination of the other two, using lance williams formula.
    """
    if l < i:
        lw1 = matrix[l,i]
    else:
        lw1 = matrix[i,l]
    if l < j:
        lw2 = matrix[l,j]
    else:
        lw2 = matrix[j,l]
    lw3 = matrix[i,j]
    return a*lw1 + b*lw2 + c*lw3 + d*abs(lw1 - lw2)
 


def agglomerative(data: list[Point], a:float, b:float, c:float, d:float, max_dist: float = 0,
                  dist: Distance = euclidean_distance) -> tuple[list[list], list[Cluster]]:
    """
    Given a set of data, the parameters for lance williams formula, the maximum distance allowed to combine two clusters,
    and the distance to use, returns the linkage matrix and the final clusters, where each list of linkage matrix is:
    [[i,j,d,n],...] with:
          i = cluster index for the first cluster which will be combined.
          j = cluster index for the second cluster which will be combined.
          d = distance between the two clusters.
          n = number of points in the new cluster.

    Parameters
    ----------
    data: set of data.
    a: parameter for lance williams.
    b: parameter for lance williams.
    c: parameter for lance williams.
    d: parameter for lance williams.
    max_dist: maximum distance allowed to combine two clusters.
    dist: distance to use.

    Returns
    -------
    linkage matrix and the list with the final clusters.
    """
    result = []
    n = len(data)
    linkage_matrix = []
    list_clusters = []
    (matrix, minimum, (i,j)) = proximity_matrix(data, dist)
    for k in range(n):
        cluster = Cluster({data[k]})
        list_clusters.append((cluster,k))
        result.append(cluster)
    for s in range(n-1):
        if s != 0:
            (minimum,(i,j)) = min_distance(matrix)
        (cluster1,k1) = list_clusters.pop(j)
        (cluster2,k2) = list_clusters[i]
        linkage_matrix.append([k1,k2,minimum,cluster1.num_points() + cluster2.num_points()])
        new_cluster = cluster1.combine(cluster2)
        list_clusters[i] = (new_cluster, n + s)
        if minimum <= max_dist:
            result.pop(j)
            result[i] = new_cluster
        for l in range(i):
            matrix[l,i] = lance_williams(matrix,l,i,j,a,b,c,d)
        for l in range(i+1,len(list_clusters)):
            if l != j:
                matrix[i,l] = lance_williams(matrix,l,i,j,a,b,c,d)
        matrix = np.delete(np.delete(matrix, j, axis = 0),j,axis = 1)

    return linkage_matrix, result
        

        
        

        
        
    
    
    
    
    
    
    
    
    
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
    
    
    
    
    
    
    
    
    
    