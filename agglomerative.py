# -*- coding: utf-8 -*-



import math
import numpy as np
import inspect
from typing import Callable

from points import Distance, Cluster, euclidean_distance, Point

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

def single(matrix: np.ndarray, l:int, i:int, j:int):
    return lance_williams(matrix, l, i, j,0.5, 0.5, 0, -0.5)

def complete(matrix: np.ndarray, l:int, i:int, j:int):
    return lance_williams(matrix, l, i, j,0.5, 0.5, 0, 0.5)

def average(matrix: np.ndarray, l:int, i:int, j:int):
    return lance_williams(matrix, l, i, j,0.5, 0.5, 0, 0)

def median(matrix: np.ndarray, l:int, i:int, j:int):
    return lance_williams(matrix, l, i, j,0.5, 0.5, -0.25, 0)

def ward(matrix: np.ndarray, l:int, i:int, j:int, n_l: int, n_i: int, n_j: int):
    denominator = n_l + n_i + n_j
    a = (n_i + n_l) / denominator
    b = (n_j + n_l) / denominator
    c = (-n_l) / denominator
    d = 0
    return lance_williams(matrix, l, i, j, a, b, c, d)

def weighted_average(matrix: np.ndarray, l:int, i:int, j:int, n_i: int, n_j: int):
    denominator = n_i + n_j
    a = n_i / denominator
    b = n_j / denominator
    c = 0
    d = 0
    return lance_williams(matrix, l, i, j, a, b, c, d)

def centroid(matrix: np.ndarray, l:int, i:int, j:int, n_i: int, n_j: int):
    denominator = n_i + n_j
    a = n_i / denominator
    b = n_j / denominator
    c = -a*b
    d = 0
    return lance_williams(matrix, l, i, j, a, b, c, d)


def agglomerative(data: list[Point], method: Callable[..., float], max_dist: float = 0,
                  dist: Distance = euclidean_distance) -> tuple[np.ndarray, list[Cluster]]:
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

    Complexity
    -------
    O(N^3) where N: number of points.

    """
    n_param = len(inspect.signature(method).parameters)
    result = []
    n = len(data)
    linkage_matrix = np.zeros((n-1,4))
    list_clusters = [0 for _ in range(n)]
    (matrix, minimum, (i,j)) = proximity_matrix(data, dist)
    for k in range(n):                             #Initial clusters
        cluster = Cluster({data[k]})
        list_clusters[k] = (cluster,k)
        result.append(cluster)
    for s in range(n-1):
        if s != 0:
            (minimum,(i,j)) = min_distance(matrix)
        (cluster1,k1) = list_clusters.pop(j)
        (cluster2,k2) = list_clusters[i]
        linkage_matrix[s] = np.array([k1,k2,minimum,cluster1.size() + cluster2.size()])
        new_cluster = cluster1.combine(cluster2)
        list_clusters[i] = (new_cluster, n + s)
        if minimum <= max_dist:                      #If two clusters combine at a distance <= max_dist, we add the combination to the final list of clusters
            result.pop(j)
            result[i] = new_cluster
        for l in range(len(list_clusters)):          #Updating distance matrix
            if l != i and l != j:
                if l < i:
                    change_pos = (l,i)
                else:
                    change_pos = (i,l)
                if n_param == 4:                                      #Different cases for not using n_l, n_i or n_l, if they're not needed.
                        matrix[change_pos] = method(matrix,l,i,j)
                else:
                    n_i = cluster1.size()
                    n_j = cluster2.size()
                    if n_param == 6:
                        matrix[change_pos] = method(matrix,l,i,j,n_i,n_j)
                    else:
                        n_l = list_clusters[l][0].size()
                        matrix[change_pos] = method(matrix,l,i,j,n_l,n_i,n_j)

        matrix = np.delete(np.delete(matrix, j, axis = 0),j,axis = 1)            #Remove row and column j
    return linkage_matrix, result
        

        
        

        
        
    
    
    
    
    
    
    
    
    
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
    
    
    
    
    
    
    
    
    
    