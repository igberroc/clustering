# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from EM import em
from metricas import silhouette_index, db_index
from points import Point, Cluster

def descompose_x_y_x_y(cluster: Cluster) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for Point in cluster.points:
       x.append(Point.coordinates[0])
       y.append(Point.coordinates[1])
    return (x,y)

def main1():
    n_clusters = 3  # Número de clusters
    n_points = 40  # Número de points por cluster
    error = 0.01
    max_iter = 100

    medias = [[0, 0],[5, 5],[-5, 5]]
    covarianzas = [[[1, 0], [0, 1]],[[1, 0.5], [0.5, 1]],[[1, -0.5], [-0.5, 1]]]
    
    data = []
    for i in range(n_clusters):
         array_points = np.random.multivariate_normal(medias[i], covarianzas[i], n_points)
         for coord in array_points:
             data.append(Point(*tuple(coord)))
    
    list_clusters = em(data,n_clusters, error, max_iter)
    silhouette = silhouette_index(list_clusters)
    print(f"El index de silhouette es: {silhouette} ")
    
    db = db_index(list_clusters)
    print(f"El index de Davies-Bouldin es: {db} ")

    (x1,y1) = descompose_x_y(list_clusters[0])
    (x2,y2) = descompose_x_y(list_clusters[1])
    (x3,y3) = descompose_x_y(list_clusters[2]) 
    plt.plot(x1,y1,'o',markerfacecolor = 'red') 
    plt.plot(x2,y2,'o',markerfacecolor = 'blue') 
    plt.plot(x3,y3,'o',markerfacecolor = 'green')


    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters mixturas gaussianas")
    plt.legend()
    plt.show()
    
    
    
 
    
    
    
    
    
    