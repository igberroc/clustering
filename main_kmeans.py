# -*- coding: utf-8 -*-


from kmeans_con_objetos import kmeans
import matplotlib.pyplot as plt
import random

from puntos import Punto, Cluster
from metricas import indice_Silueta, indice_DB



def desglosar_x_y(cluster: Cluster) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for punto in cluster.puntos:
       x.append(punto.coordenadas[0])
       y.append(punto.coordenadas[1])
    return (x,y)


def main1():
    delta = 0.25
    npuntos = 20
    datos1 = [Punto(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos2 = [Punto(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos3 = [Punto(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos4 = [Punto(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos = datos1 + datos2 + datos3 + datos4
    
    lista_clusters = kmeans(datos,4,0.001,100)
    silueta = indice_Silueta(lista_clusters)
    print(f"El indice de silueta es: {silueta} ")
    
    db = indice_DB(lista_clusters)
    print(f"El indice de Davies-Bouldin es: {db} ")

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
    
