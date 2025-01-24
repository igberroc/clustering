# -*- coding: utf-8 -*-

import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
import numpy as np

from points import Point, Cluster
from dbscan import dbscan
from metricas import indice_Silueta, indice_DB

def descompose_x_y(cluster: Cluster) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for Point in cluster.points:
       x.append(Point.coordinates[0])
       y.append(Point.coordinates[1])
    return (x,y)


def color_aleatorio():
    return f'#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}'


def main1():
    df = pd.read_csv('C:/Users/nacho/Downloads/dataset_dbscan.csv')
    lista_x = df['Weight'].tolist()
    lista_y = df['Height'].tolist()
    datos = []
    for i,j in zip(lista_x,lista_y):
        datos.append(Point(i,j))
        
    eps = 0.6
    minPts = 5
    (lista_clusters, ruido) = dbscan(datos,eps,minPts)
    
    colores_usados = set()
    for cluster in lista_clusters:
        (x,y) = descompose_x_y(cluster)
        color = color_aleatorio()
        while color in colores_usados:
            color = color_aleatorio()
        colores_usados.add(color)
        plt.scatter(x,y, s = 10, color = color)
    
    silueta = indice_Silueta(lista_clusters)
    print(f"El indice de silueta es: {silueta} ")
    
    db = indice_DB(lista_clusters)
    print(f"El indice de Davies-Bouldin es: {db} ")

    
    (x,y) = descompose_x_y(ruido)
    plt.scatter(x,y, s = 10, color = 'black')
    
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters DBSCAN")
    plt.legend()
    plt.show()
    
def main2():
    data, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    datos = []
    for [i,j] in data:
        datos.append(Point(i,j))
    eps = 0.2
    minPts = 5
    (lista_clusters, ruido) = dbscan(datos,eps,minPts)
    
    colores_usados = set()
    for cluster in lista_clusters:
        (x,y) = descompose_x_y(cluster)
        color = color_aleatorio()
        while color in colores_usados:
            color = color_aleatorio()
        colores_usados.add(color)
        plt.scatter(x,y, s = 10, color = color)
    
    silueta = indice_Silueta(lista_clusters)
    print(f"El indice de silueta es: {silueta} ")
    
    db = indice_DB(lista_clusters)
    print(f"El indice de Davies-Bouldin es: {db} ")

    (x,y) = descompose_x_y(ruido)
    plt.plot(x,y, 'o', color = 'black')
    
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters DBSCAN")
    plt.legend()
    plt.show()
    

def main3():
    
    def twospirals(n_points, ruido=.5):
        epsilon = 0.1
        n = (np.random.rand(n_points,1)+epsilon) * 780 * (2*np.pi)/360
        d1x = -np.cos(n)*n + np.random.rand(n_points,1) * ruido
        d1y = np.sin(n)*n + np.random.rand(n_points,1) * ruido
        C_1 = np.hstack((d1x,d1y))
        C_2 = np.hstack((-d1x,-d1y))
        return np.vstack((C_1, C_2))
    
    n_points = 500
    dataset = twospirals(n_points)
    datos = []
    for [x1,x2] in dataset:
        datos.append(Point(x1,x2))
    eps = 1.7
    minPts = 2
    (lista_clusters, ruido) = dbscan(datos,eps,minPts)
    
    colores_usados = set()
    for cluster in lista_clusters:
        (x,y) = descompose_x_y(cluster)
        color = color_aleatorio()
        while color in colores_usados:
            color = color_aleatorio()
        colores_usados.add(color)
        plt.scatter(x,y, s = 10, color = color)
    
    silueta = indice_Silueta(lista_clusters)
    print(f"El indice de silueta es: {silueta} ")
    
    db = indice_DB(lista_clusters)
    print(f"El indice de Davies-Bouldin es: {db} ")


    (x,y) = descompose_x_y(ruido)
    plt.plot(x,y, 'o', color = 'black')
    
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters DBSCAN")
    plt.legend()
    plt.show()
        
        
   



    



