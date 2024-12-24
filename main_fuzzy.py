# -*- coding: utf-8 -*-

import seaborn as sns
import random
import matplotlib.pyplot as plt

from puntos import Punto
from fuzzy import fuzzy_cmeans

def desglosar_x_y(lista: list[Punto]) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for punto in lista:
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
    (x,y) = desglosar_x_y(datos)
    
    inicial_centroides = [Punto(random.uniform(-5,5),random.uniform(-5,5)) for _ in range(4)]
    matriz_pertenencia = fuzzy_cmeans(datos,inicial_centroides,2,4,0.001, 100)
    
    
    for i in range(4):
        plt.figure()
        scatter = plt.scatter(x,y, c = matriz_pertenencia[i,:], cmap=plt.cm.Blues, s = 50)
        plt.colorbar(scatter, label=f'Grado de pertenencia al cluster {i+1}')
        plt.title(f'Cluster {i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    
def main_penguins():
    penguins = sns.load_dataset('penguins').dropna()
    X = penguins[['bill_length_mm', 'bill_depth_mm']].values.tolist()
    for i in range(len(X)):
        X[i] = Punto(X[i][0],X[i][1])
        
    inicial_centroides = [Punto(random.uniform(-5,5),random.uniform(-5,5)) for _ in range(3)]
    matriz_pertenencia = fuzzy_cmeans(X,inicial_centroides,2,3,0.001, 100)
    
    (x,y) = desglosar_x_y(X)
    for i in range(3):
        plt.figure()
        scatter = plt.scatter(x,y, c = matriz_pertenencia[i,:], cmap=plt.cm.Blues, s = 50)
        plt.colorbar(scatter, label=f'Grado de pertenencia al cluster {i+1}')
        plt.title(f'Cluster {i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    

