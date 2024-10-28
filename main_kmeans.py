# -*- coding: utf-8 -*-

from typing import List
from kmeans_con_objetos import kmeans
import matplotlib.pyplot as plt
from puntos import Punto2D, distancia_euclidea, Clusters
import random


def desglosar_x_y(cluster: Clusters) -> (List[Punto2D], List[Punto2D]):
    x = []
    y = []
    for punto in cluster.puntos:
       x.append(punto.x)
       y.append(punto.y)
    return (x,y)


def main():
    delta = 0.25
    npuntos = 20
    datos1 = [Punto2D(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos2 = [Punto2D(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos3 = [Punto2D(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos4 = [Punto2D(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos = datos1 + datos2 + datos3 + datos4
    
    euclidea = distancia_euclidea()
    lista_clusters = kmeans(datos,4,0.001,100, euclidea)

    (x1,y1) = desglosar_x_y(lista_clusters[0])
    (x2,y2) = desglosar_x_y(lista_clusters[1])
    (x3,y3) = desglosar_x_y(lista_clusters[2])
    (x4,y4) = desglosar_x_y(lista_clusters[3])  
    plt.plot(x1,y1,'o',markerfacecolor = 'red') 
    plt.plot(x2,y2,'o',markerfacecolor = 'blue') 
    plt.plot(x3,y3,'o',markerfacecolor = 'green')
    plt.plot(x4,y4,'o',markerfacecolor = 'yellow')
    
    plt.xlim(-0.5,1.5)
    plt.ylim(-0.5,1.5)
    
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters k-means")
    plt.legend()
    plt.show()
    
    
