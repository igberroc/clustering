# -*- coding: utf-8 -*-



import random

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


from aglomerativo import aglomerativo
from points import Point
from metricas import silhouette_index, db_index

def main1():
    delta = 0.25
    npoints = 20
    data1 = [Point(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data2 = [Point(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data3 = [Point(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data4 = [Point(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data = data1 + data2 + data3 + data4
    
    (matriz_enlace,resultado) = aglomerativo(data,0.5,0.5,0,-0.5,0.4)
    silhouette = silhouette_index(resultado)
    print(f"El index de silhouette es: {silhouette} ")
    
    
    db = db_index(resultado)
    print(f"El index de Davies-Bouldin es: {db} ")
    
    plt.figure()
    dendrogram(np.array(matriz_enlace))
    plt.xlabel("index de los clusters")
    plt.ylabel("Distance entre clusters")
    plt.show()


def main2():
    iris = load_iris()
    lista_iris = (iris.data).tolist()
    data = []
    for dato in lista_iris:
        p = Point()
        p.coordinates = tuple(dato)
        data.append(p)
    (matriz_enlace,resultado) = aglomerativo(data,0.5,0.5,0,-0.5,1)
    
    silhouette = silhouette_index(resultado)
    print(f"El index de silhouette es: {silhouette} ")
    db = db_index(resultado)
    print(f"El index de Davies-Bouldin es: {db} ")
    
    
    plt.figure()
    dendrogram(np.array(matriz_enlace))
    plt.xlabel("index de los clusters")
    plt.ylabel("Distance entre clusters")
    plt.show()

    

