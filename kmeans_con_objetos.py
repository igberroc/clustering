# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:07:13 2024

@author: nacho
"""

from typing import List
from puntos import Distancia, Cluster, distancia_euclidea, Punto
import random
import copy


def diferencia_centroides(centroides, new_centroides,error, dist: Distancia = distancia_euclidea) -> bool:
    if centroides == []:
        return True
    for i in range(len(centroides)):
        if dist(centroides[i], new_centroides[i]) > error:
            return True
    return False


def kmeans(datos: List[Punto], n_clusters: int, error: float, max_iterac: int, 
           dist: Distancia = distancia_euclidea) -> List[Cluster]:

    dimension = datos[0].dimension()
    iterac = 0
    lista_clusters = [Cluster() for _ in range(n_clusters)]
    centroides = []
    new_centroides = random.sample(datos,n_clusters)
    while iterac < max_iterac and diferencia_centroides(centroides, new_centroides,error, dist):
        for cluster in lista_clusters:
            cluster.vaciar()
        centroides = copy.deepcopy(new_centroides)
        new_centroides = []
        for dato in datos:
            L = []
            for centroide in centroides:
                L.append(dist(dato, centroide))
            minimo = min(L)
            posicion_minimo = L.index(minimo)
            cluster = lista_clusters[posicion_minimo]
            cluster.agregar_punto(dato)
        for cluster in lista_clusters:
            centroide = cluster.calcular_centroides(dimension)
            new_centroides.append(centroide)
        iterac += 1
    return lista_clusters

