# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:20:09 2024

@author: nacho
"""

import math
from typing import Self, Callable, TypeVar


class Punto:
    def __init__(self, *coordenadas: float):
        self.coordenadas = coordenadas
    
    def dimension(self) -> int:
        return len(self.coordenadas)
    
    def get_coordenadas(self) -> tuple():
        return self.coordenadas
    
    def sumar(self,other: Self) -> Self:
        suma = [0 for _ in range(self.dimension())]
        for i in range(self.dimension()):
            suma[i] = self.coordenadas[i] + other.coordenadas[i]
        return Punto(*tuple(suma))
    
    def mul_num(self, n:int) -> Self:
        coord = list(self.coordenadas)
        for i in range(self.dimension()):
            coord[i] = coord[i] * n
        return Punto(*tuple(coord))
            
    def dividir_num(self, n:int) -> Self:
        coord = list(self.coordenadas)
        for i in range(self.dimension()):
            coord[i] = coord[i]/n
        return Punto(*tuple(coord))
    
    @staticmethod
    def punto_nulo(dimension) -> Self:
        p = Punto()
        p.coordenadas = (0,)*dimension
        return p
    
    
T = TypeVar('T')
type Distancia = Callable[[T,T], float]   

     
def distancia_euclidea(punto1: Punto, punto2: Punto) -> float:
    cord1 = punto1.get_coordenadas()
    cord2 = punto2.get_coordenadas()
    d = 0
    for x1,y1 in zip(cord1,cord2):
        d += (x1 - y1)**2
    return math.sqrt(d)
        
        
def distancia_manattan(punto1: Punto, punto2: Punto) -> float:
    cord1 = punto1.get_coordenadas()
    cord2 = punto2.get_coordenadas()
    d = 0
    for x1,y1 in zip(cord1,cord2):
        d += abs(x1 - y1)
    return d


def distancia_str(s1: str, s2: str) -> float:
    return sum(1 for x,y in zip(s1,s2) if x != y)


class Cluster:
    def __init__(self, puntos: set[Punto] = None):
        if puntos == None:
            self.puntos = set()
        else:
            self.puntos = puntos
        
    def agregar_punto(self,punto: Punto):
        self.puntos.add(punto)
        
    def quitar_punto(self,punto: Punto):
        self.puntos.remove(punto)
        
    def calcular_centroide(self) -> Punto:
        dimension = list(self.puntos)[0].dimension()
        suma = Punto.punto_nulo(dimension)
        for punto in self.puntos:
            suma = suma.sumar(punto)   
        n = len(self.puntos)
        suma = suma.dividir_num(n)
        return suma
    
    def vaciar(self) -> None:
        self.puntos = set()
    
    def num_puntos(self) -> int:
        return len(self.puntos)
    
    def copiar(self) -> Self:
        cluster = Cluster()
        cluster.puntos = self.puntos.copy()
        return cluster
    
    def tamaño(self) -> int:
        return (len(self.puntos))
    
    def combinar(self,other) -> Self:
        cluster = self.copiar()
        cluster.puntos = cluster.puntos | other.puntos
        return cluster
        

def desglosar_x_y(cluster: Cluster) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for punto in cluster.puntos:
       x.append(punto.coordenadas[0])
       y.append(punto.coordenadas[1])
    return (x,y)









  