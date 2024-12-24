# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from EM import em
from metricas import indice_Silueta, indice_DB
from puntos import Punto, Cluster

def desglosar_x_y(cluster: Cluster) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for punto in cluster.puntos:
       x.append(punto.coordenadas[0])
       y.append(punto.coordenadas[1])
    return (x,y)

def main1():
    n_clusters = 3  # Número de clusters
    n_points = 40  # Número de puntos por cluster
    error = 0.01
    max_iterac = 100

    medias = [[0, 0],[5, 5],[-5, 5]]
    covarianzas = [[[1, 0], [0, 1]],[[1, 0.5], [0.5, 1]],[[1, -0.5], [-0.5, 1]]]
    
    datos = []
    for i in range(n_clusters):
         array_puntos = np.random.multivariate_normal(medias[i], covarianzas[i], n_points)
         for coord in array_puntos:
             datos.append(Punto(*tuple(coord)))
    
    lista_clusters = em(datos,n_clusters, error, max_iterac)
    silueta = indice_Silueta(lista_clusters)
    print(f"El indice de silueta es: {silueta} ")
    
    db = indice_DB(lista_clusters)
    print(f"El indice de Davies-Bouldin es: {db} ")

    (x1,y1) = desglosar_x_y(lista_clusters[0])
    (x2,y2) = desglosar_x_y(lista_clusters[1])
    (x3,y3) = desglosar_x_y(lista_clusters[2]) 
    plt.plot(x1,y1,'o',markerfacecolor = 'red') 
    plt.plot(x2,y2,'o',markerfacecolor = 'blue') 
    plt.plot(x3,y3,'o',markerfacecolor = 'green')


    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters mixturas gaussianas")
    plt.legend()
    plt.show()
    
    
    
 
    
    
    
    
    
    