# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:04:24 2024

@author: nacho
"""
import random

from sklearn.datasets import load_iris
from aglomerativo import aglomerativo
from puntos import Punto, distancia_euclidea

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def main2():
    iris = load_iris()
    lista_iris = (iris.data).tolist()
    datos = []
    for dato in lista_iris:
        p = Punto()
        p.coordenadas = tuple(dato)
        datos.append(p)
    
    euclidea = distancia_euclidea()
    matriz_enlace = aglomerativo(datos,0.5,0.5,0,-0.5,euclidea)
    euclidea = distancia_euclidea()
    matriz_enlace = aglomerativo(datos,0.5,0.5,0,-0.5,euclidea)
    print(matriz_enlace)
    plt.figure()
    dendrogram(np.array(matriz_enlace))
    plt.xlabel("Indice de los clusters")
    plt.ylabel("Distancia entre clusters")
    plt.show()

    

def main4():
    datos = []
    for _ in range(20):
        datos.append(Punto(random.randint(1,10),random.randint(1,10)))
    
    euclidea = distancia_euclidea()
    matriz_enlace = aglomerativo(datos,0.5,0.5,0,-0.5,euclidea)
    print(matriz_enlace)
    plt.figure()
    dendrogram(np.array(matriz_enlace))
    plt.xlabel("Indice de los clusters")
    plt.ylabel("Distancia entre clusters")
    plt.show()


