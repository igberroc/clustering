# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:20:09 2024

@author: nacho
"""
import copy
from abc import ABC, abstractmethod
import math

class Punto2D:
    def __init__(self, x: float, y:float):
        self.x = x
        self.y = y
        
    def get_coordenadas(self):
        return self.x, self.y


class Distancia(ABC):
    @abstractmethod
    def d(self, punto1, punto2):
        pass
    

class distancia_euclidea(Distancia):
    def d(self, punto1: Punto2D, punto2: Punto2D) -> float:
        x1, y1 = punto1.get_coordenadas()
        x2, y2 = punto2.get_coordenadas()
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        
class distancia_manhattan(Distancia):
    def d(self, punto1: Punto2D, punto2: Punto2D) -> float:
        x1, y1 = punto1.get_coordenadas()
        x2, y2 = punto2.get_coordenadas()
        return abs(x1 - x2) + abs(y1 - y2)

class Clusters:
    def __init__(self):
        self.puntos = set()
    
    def agregar_punto(self, punto: Punto2D):
        self.puntos.add(punto)
        
    def calcular_centroides(self) -> Punto2D:
        suma_x = sum(punto.x for punto in self.puntos)
        suma_y = sum(punto.y for punto in self.puntos)
        n = len(self.puntos)
        return Punto2D(suma_x/n, suma_y/n)
    
    def vaciar(self):
        self.puntos = set()
    
    def copiar(self):
        cluster = Clusters()
        cluster.puntos = self.puntos.copy()
        return cluster
    
    def dividir(self):
        cluster1 = self.copiar()
        cluster2 = Clusters()
        cluster2.puntos = set([cluster1.puntos.pop()])
        return cluster1, cluster2
        
    def tamaño(self) -> int:
        return (len(self.puntos))
    
    def combinar(self,other):
        cluster = self.copiar()
        cluster.puntos = cluster.puntos | other.puntos
        return cluster
        
class Lance_Williams(ABC):
    def d_clusters(self, cluster1: Clusters, cluster2: Clusters, a: float, b: float, c: float, d:float , metodo: 'Distancia') -> float:
        if cluster1.tamaño() == 1 and cluster2.tamaño() == 1:
            return metodo.d(list(cluster1.puntos)[0], list(cluster2.puntos)[0])
        if cluster1.tamaño() != 1 and cluster2.tamaño() == 1:
            return self.d_clusters(cluster2, cluster1,a,b,c,d,metodo)
        else:
            division1,division2 = cluster2.dividir()
            A = self.d_clusters(cluster1, division1,a,b,c,d,metodo)
            B = self.d_clusters(cluster1, division2,a,b,c,d,metodo)
            C = self.d_clusters(division1, division2,a,b,c,d,metodo)
            D = abs(self.d_clusters(cluster1, division1,a,b,c,d,metodo) - self.d_clusters(cluster1, division2,a,b,c,d,metodo))
            return a*A + b*B + c*C + d*D
        
    
cluster1 = Clusters()
cluster2 = Clusters()
cluster1.agregar_punto(Punto2D(1,0))
cluster1.agregar_punto(Punto2D(-1,0))        
cluster2.agregar_punto(Punto2D(2,0))      
cluster2.agregar_punto(Punto2D(3,1))       
cluster2.agregar_punto(Punto2D(3,4)) 
lance = Lance_Williams()
euclidea = distancia_euclidea()
a,b,c,d = 0.5,0.5,0,0.5
print(lance.d_clusters(cluster1, cluster2, a, b, c, d, euclidea))









  