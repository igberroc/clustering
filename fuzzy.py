# -*- coding: utf-8 -*-

import numpy as np
import copy

from puntos import Distancia, distancia_euclidea, Punto


def diferencia_centroides(centroides: list[Punto], new_centroides: list[Punto], 
                          error: int, dist: Distancia = distancia_euclidea) -> bool:
    """
    Given the list of new and old centroids, it returns true if stop condition is satisfied 
    or false if not.

    Parameters
    ----------
    centroides : old centroids
    new_centroides : new centroids
    error : maximum error for new and old centroids.
    dist : distancia to use

    Returns
    -------
    bool: stop condition satisfied or not

    """
    if centroides == []:
        return True
    for i in range(len(centroides)):
        if dist(centroides[i], new_centroides[i]) > error:
            return True
    return False


def fuzzy_cmeans(datos: list[Punto], inicial_centroides: list[Punto], m: int, c: int, error: float, max_iterac: int, 
           dist: Distancia = distancia_euclidea) -> np.ndarray:
    """
    Given a set of data and parameters, returns the fuzzy partition matrix.
    Parameters
    ----------
    datos : list with data
    inicial_centroides : list with initial centroids
    m : fuzzification parameter (usually set to 2)
    c : number of clusters
    error : maximum error for new and old centroids.
    max_iterac : maximum number of iterations
    dist : distance to use

    Returns
    -------
    matriz_pertenencia : fuzzy partition matrix

    """
    n = len(datos)
    iterac = 0
    centroides = []
    new_centroides = inicial_centroides
    matriz_pertenencia = np.zeros((c,n))
    while iterac < max_iterac and diferencia_centroides(centroides, new_centroides, error, dist):
        centroides = copy.deepcopy(new_centroides)
        new_centroides = [0 for _ in range(c)]
        for j in range(n):
            for i in range(c):
                sumatorio = 0
                for l in range(c):
                    coef = dist(datos[j], centroides[l]) / dist(datos[j],centroides[i])
                    coef = coef ** (1/(1 - m))
                    sumatorio += coef
                matriz_pertenencia[i,j] = 1 / sumatorio
        for i in range(c):
            div = 0
            for j in range(n):
                punto = datos[j]
                u = matriz_pertenencia[i,j] ** m
                if j == 0:
                    suma_puntos = punto.mul_num(u)
                else:
                    suma_puntos = suma_puntos.sumar(punto.mul_num(u))
                    div += u
            new_centroides[i] = suma_puntos.dividir_num(div)
        iterac += 1
    return matriz_pertenencia
        
        
                
            
        
        
                
                
                
            
            
        
    