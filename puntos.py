# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:20:09 2024

@author: nacho
"""

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
        self.puntos = []
    
    def agregar_punto(self, punto: Punto2D):
        self.puntos.append(punto)
        
    def calcular_centroides(self) -> Punto2D:
        suma_x = sum(punto.x for punto in self.puntos)
        suma_y = sum(punto.y for punto in self.puntos)
        n = len(self.puntos)
        return Punto2D(suma_x/n, suma_y/n)
    
    def vaciar(self):
        self.puntos = []