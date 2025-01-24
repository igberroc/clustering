# -*- coding: utf-8 -*-



import random

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


from aglomerativo import aglomerativo
from points import Point
from metricas import indice_Silueta, indice_DB

def main1():
    delta = 0.25
    npoints = 20
    datos1 = [Point(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    datos2 = [Point(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    datos3 = [Point(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    datos4 = [Point(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    datos = datos1 + datos2 + datos3 + datos4
    
    (matriz_enlace,resultado) = aglomerativo(datos,0.5,0.5,0,-0.5,0.4)
    silueta = indice_Silueta(resultado)
    print(f"El indice de silueta es: {silueta} ")
    
    
    db = indice_DB(resultado)
    print(f"El indice de Davies-Bouldin es: {db} ")
    
    plt.figure()
    dendrogram(np.array(matriz_enlace))
    plt.xlabel("Indice de los clusters")
    plt.ylabel("Distance entre clusters")
    plt.show()


def main2():
    iris = load_iris()
    lista_iris = (iris.data).tolist()
    datos = []
    for dato in lista_iris:
        p = Point()
        p.coordinates = tuple(dato)
        datos.append(p)
    (matriz_enlace,resultado) = aglomerativo(datos,0.5,0.5,0,-0.5,1)
    
    silueta = indice_Silueta(resultado)
    print(f"El indice de silueta es: {silueta} ")
    db = indice_DB(resultado)
    print(f"El indice de Davies-Bouldin es: {db} ")
    
    
    plt.figure()
    dendrogram(np.array(matriz_enlace))
    plt.xlabel("Indice de los clusters")
    plt.ylabel("Distance entre clusters")
    plt.show()

    

