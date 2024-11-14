# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:04:24 2024

@author: nacho
"""
import random

from sklearn.datasets import load_iris
from aglomerativo import aglomerativo
from puntos import Punto

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def main1():
    iris = load_iris()
    lista_iris = (iris.data).tolist()
    datos = []
    for dato in lista_iris:
        p = Punto()
        p.coordenadas = tuple(dato)
        datos.append(p)
    matriz_enlace = aglomerativo(datos,0.5,0.5,0,-0.5)
    plt.figure()
    dendrogram(np.array(matriz_enlace))
    plt.xlabel("Indice de los clusters")
    plt.ylabel("Distancia entre clusters")
    plt.show()

    

def main2():
    datos = []
    for _ in range(20):
        datos.append(Punto(random.randint(1,10),random.randint(1,10)))
    matriz_enlace = aglomerativo(datos,0.5,0.5,0,-0.5)
    plt.figure()
    dendrogram(np.array(matriz_enlace))
    plt.xlabel("Indice de los clusters")
    plt.ylabel("Distancia entre clusters")
    plt.show()

def main3():
    delta = 0.25
    npuntos = 20
    datos1 = [Punto(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos2 = [Punto(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos3 = [Punto(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos4 = [Punto(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npuntos)]
    datos = datos1 + datos2 + datos3 + datos4
    
    matriz_enlace = aglomerativo(datos,0.5,0.5,0,-0.5)
    plt.figure()
    dendrogram(np.array(matriz_enlace))
    plt.xlabel("Indice de los clusters")
    plt.ylabel("Distancia entre clusters")
    plt.show()
    