# -*- coding: utf-8 -*-

import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
import numpy as np

from puntos import Punto, Cluster
from dbscan import dbscan


def desglosar_x_y(cluster: Cluster) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for punto in cluster.puntos:
       x.append(punto.coordenadas[0])
       y.append(punto.coordenadas[1])
    return (x,y)


def color_aleatorio():
    return f'#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}'


def main1():
    delta = 0.25
    npuntos = 20
    datos1 = [Punto(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos2 = [Punto(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos3 = [Punto(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos4 = [Punto(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos = datos1 + datos2 + datos3 + datos4
    
    eps = 0.25
    minPts = 6
    (lista_clusters, ruido) = dbscan(datos,eps,minPts)
    
    colores_usados = set()
    for cluster in lista_clusters:
        (x,y) = desglosar_x_y(cluster)
        color = color_aleatorio()
        while color in colores_usados:
            color = color_aleatorio()
        colores_usados.add(color)
        plt.plot(x,y,'o', color = color)
    
    (x,y) = desglosar_x_y(ruido)
    plt.plot(x,y,'o', color = 'black')
    
    
    plt.xlim(-0.5,1.5)
    plt.ylim(-0.5,1.5)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters k-means")
    plt.legend()
    plt.show()
        

def main2():
    df = pd.read_csv('C:/Users/nacho/Downloads/dataset_dbscan.csv')
    lista_x = df['Weight'].tolist()
    lista_y = df['Height'].tolist()
    datos = []
    for i,j in zip(lista_x,lista_y):
        datos.append(Punto(i,j))
        
    eps = 0.6
    minPts = 5
    (lista_clusters, ruido) = dbscan(datos,eps,minPts)
    
    colores_usados = set()
    for cluster in lista_clusters:
        (x,y) = desglosar_x_y(cluster)
        color = color_aleatorio()
        while color in colores_usados:
            color = color_aleatorio()
        colores_usados.add(color)
        plt.scatter(x,y, s = 10, color = color)
    
    (x,y) = desglosar_x_y(ruido)
    plt.scatter(x,y, s = 10, color = 'black')
    
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters k-means")
    plt.legend()
    plt.show()
    
def main3():
    data, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    lista_x = data[:,0].tolist()
    lista_y = data[:,1].tolist()
    datos = []
    for i,j in zip(lista_x,lista_y):
        datos.append(Punto(i,j))
        
    eps = 0.2
    minPts = 5
    (lista_clusters, ruido) = dbscan(datos,eps,minPts)
    
    colores_usados = set()
    for cluster in lista_clusters:
        (x,y) = desglosar_x_y(cluster)
        color = color_aleatorio()
        while color in colores_usados:
            color = color_aleatorio()
        colores_usados.add(color)
        plt.scatter(x,y, s = 10, color = color)
    
    (x,y) = desglosar_x_y(ruido)
    plt.plot(x,y, 'o', color = 'black')
    
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters k-means")
    plt.legend()
    plt.show()
    
    
    
