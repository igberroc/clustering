# -*- coding: utf-8 -*-

import numpy as np
import copy

from points import Distance, euclidean_distance, Point


def centroids_condition(centroids: list[Point], new_centroids: list[Point], 
                          error: int, dist: Distance = euclidean_distance) -> bool:
    """
    Given the list of new and old centroids, it returns true if stop condition is satisfied 
    or false if not.

    Parameters
    ----------
    centroids : old centroids
    new_centroids : new centroids
    error : maximum error for new and old centroids.
    dist : Distance to use

    Returns
    -------
    bool: stop condition satisfied or not

    """
    if centroids == []:
        return True
    for i in range(len(centroids)):
        if dist(centroids[i], new_centroids[i]) > error:
            return True
    return False


def fuzzy_cmeans(data: list[Point], inicial_centroids: list[Point], m: int, c: int, error: float, max_iter: int, 
           dist: Distance = euclidean_distance) -> np.ndarray:
    """
    Given a set of data and parameters, returns the fuzzy partition matrix.
    Parameters
    ----------
    data : list with data
    inicial_centroids : list with initial centroids
    m : fuzzification parameter (usually set to 2)
    c : number of clusters
    error : maximum error for new and old centroids.
    max_iter : maximum number of iterations
    dist : distance to use

    Returns
    -------
    matriz_pertenencia : fuzzy partition matrix

    """
    n = len(data)
    iterac = 0
    centroids = []
    new_centroids = inicial_centroids
    matriz_pertenencia = np.zeros((c,n))
    while iterac < max_iter and centroids_condition(centroids, new_centroids, error, dist):
        centroids = copy.deepcopy(new_centroids)
        new_centroids = [0 for _ in range(c)]
        for j in range(n):
            for i in range(c):
                sumatorio = 0
                for l in range(c):
                    coef = dist(data[j], centroids[l]) / dist(data[j],centroids[i])
                    coef = coef ** (1/(1 - m))
                    sumatorio += coef
                matriz_pertenencia[i,j] = 1 / sumatorio
        for i in range(c):
            div = 0
            for j in range(n):
                Point = data[j]
                u = matriz_pertenencia[i,j] ** m
                if j == 0:
                    suma_points = Point.mul_num(u)
                else:
                    suma_points = suma_points.sumar(Point.mul_num(u))
                    div += u
            new_centroids[i] = suma_points.dividir_num(div)
        iterac += 1
    return matriz_pertenencia
        
        
                
            
        
        
                
                
                
            
            
        
    