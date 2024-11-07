# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:20:09 2024

@author: nacho
"""

from abc import ABC, abstractmethod
import math

class Punto:
    def __init__(self, *coordenadas):
        self.coordenadas = coordenadas
    
    def dimension(self):
        return len(self.coordenadas)
    
    def get_coordenadas(self):
        return self.coordenadas
    
    def sumar(self,other):
        suma = [0 for _ in range(self.dimension())]
        for i in range(self.dimension()):
            suma[i] = self.coordenadas[i] + other.coordenadas[i]
        self.coordenadas = tuple(suma)
            
    def dividir_num(self,n):
        coord = list(self.coordenadas)
        for i in range(self.dimension()):
            coord[i] = coord[i]/n
        self.coordenadas = tuple(coord)
    
    @staticmethod
    def punto_nulo(dimension):
        p = Punto()
        p.coordenadas = (0,)*dimension
        return p
        
class Distancia(ABC):
    @abstractmethod
    def d(self, punto1, punto2):
        pass
    

class distancia_euclidea(Distancia):
    def d(self, punto1: Punto, punto2: Punto) -> float:
        cord1 = punto1.get_coordenadas()
        cord2 = punto2.get_coordenadas()
        d = 0
        for x1,y1 in zip(cord1,cord2):
            d += (x1 - y1)**2
        return math.sqrt(d)
        
        
class distancia_manhattan(Distancia):
    def d(self, punto1: Punto, punto2: Punto) -> float:
        cord1 = punto1.get_coordenadas()
        cord2 = punto2.get_coordenadas()
        d = 0
        for x1,y1 in zip(cord1,cord2):
            d += abs(x1 - y1)
        return d

class Clusters:
    def __init__(self):
        self.puntos = set()
    
    def agregar_punto(self, punto: Punto):
        self.puntos.add(punto)
        
    def calcular_centroides(self,dimension: int) -> Punto:
        suma = Punto.punto_nulo(dimension)
        for punto in self.puntos:
            suma.sumar(punto)   
        n = len(self.puntos)
        suma.dividir_num(n)
        return suma
    
    def vaciar(self):
        self.puntos = set()
    
    def num_puntos(self):
        return len(self.puntos)
    
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
            D = abs(A - B)
            return a*A + b*B + c*C + d*D
        
    







  