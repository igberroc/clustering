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


def update_matrix(matrix: np.ndarray, i: int, j: int, method: Callable[..., float],
                    n_param: int, list_clusters: list[Cluster]) -> None:
    """
    Given the proximity matrix, the indexes of the clusters which will be combined, the linkage
    method, the number of parameters for the Lance-Williams formula, and the list of clusters, updates
    the proximity matrix after both cluster's combination.

    Parameters
    ----------
    matrix: proximity matrix.
    i: index of one of the clusters that is going to be combined.
    j: index of the other cluster that is going to be combined.
    method: linkage method.
    n_param: number of parameters for the Lance-Williams formula.
    list_clusters: list of current clusters.

    """
    for l in range(len(list_clusters)):
        if l != i and l != j:
            if l < i:
                change_pos = (l, i)
            else:
                change_pos = (i, l)
            if n_param == 4:  # Different cases for not using n_l, n_i or n_j, if they're not needed.
                matrix[change_pos] = method(matrix, l, i, j)
            else:
                n_i = list_clusters[i].size()
                n_j = list_clusters[j].size()
                if n_param == 6:
                    matrix[change_pos] = method(matrix, l, i, j, n_i, n_j)
                else:
                    n_l = list_clusters[l].size()
                    matrix[change_pos] = method(matrix, l, i, j, n_l, n_i, n_j)


def agglomerative(data: list[Point], method: Callable[..., float], max_dist: float = 0,
                  dist: Distance = euclidean_distance) -> tuple[np.ndarray, list[Cluster]]:
    """
    Given a set of data, the linkage method, the maximum distance allowed to combine two clusters,
    and the distance to use, returns the linkage matrix and the final clusters, where each list of linkage matrix is:
    [[i,j,d,n],...] with:
          i = cluster index for the first cluster which will be combined.
          j = cluster index for the second cluster which will be combined.
          d = distance between the two clusters.
          n = number of points in the new cluster.

    Parameters
    ----------
    data: set of data.
    method: linkage method.
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
    n = len(data)
    linkage_matrix = np.zeros((n-1,4))  #Linkage matrix for the future dendrogram.
    list_clusters = []
    clusters_indexes = []      #Cluster numbers for the linkage matrix.
    result = []
    for k in range(n):                             #Initial clusters
        cluster = Cluster({data[k]})
        list_clusters.append(cluster)
        result.append(cluster)
        clusters_indexes.append(k)
    last_merge_valid = False          #True if the clusters in the last iteration were combined at a distance <= max_dist, and False otherwise.
    (matrix, minimum, (i,j)) = proximity_matrix(data, dist)  #Initial proximity matrix, with minimum distance between clusters and their indexes.
    for s in range(n-1):
        if s != 0:
            (minimum,(i,j)) = min_distance(matrix)
        if minimum <= max_dist:
            last_merge_valid = True
        elif last_merge_valid:        #If minimum > max_dist and the last merge was at a distance <= max_dist, the result will be the last list of clusters. We do this
            result = list_clusters.copy()   # to avoid copying list_clusters every time minimum <= max_dist, and only do it when the last merge occurred but doesn't happen now.
            last_merge_valid = False      #Variable updated.

        update_matrix(matrix, i, j, method, n_param, list_clusters)
        matrix = np.delete(np.delete(matrix, j, axis=0), j, axis=1)  # Remove row and column j

        cluster1 = list_clusters[i]
        cluster2 = list_clusters.pop(j)   #We can do this because i < j, so the element at position i remains unchanged.
        k1 = clusters_indexes[i]
        k2 = clusters_indexes.pop(j)
        linkage_matrix[s] = np.array([k1,k2,minimum,cluster1.size() + cluster2.size()])  #New linkage info.

        new_cluster = cluster1.combine(cluster2)
        list_clusters[i] = new_cluster      #We update the list of clusters.
        clusters_indexes[i] = n + s         #We update the cluster index.
    if last_merge_valid:
        result = list_clusters    #Copy not needed, because list_clusters won`t be modified.
    return linkage_matrix, result



        

        
        
    
    
    
    
    
    
    
    
    
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
    
    
    
    
    
    
    
    
    
    