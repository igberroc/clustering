# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:42:14 2024

@author: Nacho
"""
from typing import List
import math
import numpy as np

from puntos import Distancia, Cluster, distancia_euclidea, Punto

def crear_matriz_proximidad(datos:List[Punto], dist: Distancia = distancia_euclidea) -> (np.ndarray,float,(int,int)):
    n = len(datos)
    M = np.zeros((n,n))
    minimo = math.inf
    i_minimo = 0
    j_minimo = 1
    for i in range(n-1):
        for j in range(i+1,n):
            distancia = dist(datos[i],datos[j])
            M[i,j] = distancia
            if distancia < minimo:
                i_minimo = i
                j_minimo = j
                minimo = distancia
    return (M,minimo,(i_minimo,j_minimo))
    
def minima_distancia(M: np.ndarray) -> (float,(int,int)):
    n = len(M)
    d_minima = math.inf
    i_minimo = 0
    j_minimo = 1
    for i in range(n-1):
        for j in range(i+1,n):
            if M[i,j] < d_minima:
                i_minimo = i
                j_minimo = j
                d_minima = M[i,j]
    return(d_minima,(i_minimo,j_minimo))
            
def lance_williams(M:np.ndarray, l:int, i:int, j:int, a:float, b:float, c:float, d:float) -> float:
    A = M[l,i]
    B = M[l,j]
    C = M[i,j]
    return a*A + b*B + c*C + d*abs(A - B)
 

#Damos la maxima distancia a la que queremos que se unan los cluster
def aglomerativo(datos: List[Punto], a:float, b:float, c:float, d:float, maximo: float = 0,
                 dist: Distancia = distancia_euclidea) -> (List, List[Cluster]):
    resultado = []
    n = len(datos)
    matriz_enlace = []
    lista_clusters = []
    (M,d_minima,(i,j)) = crear_matriz_proximidad(datos,dist)
    for k in range(n):
        cluster = Cluster({datos[k]})
        lista_clusters.append((cluster,k))
        resultado.append(cluster)
    for s in range(n-1):
        if s != 0:
            (d_minima,(i,j)) = minima_distancia(M)
        (cluster1,k1) = lista_clusters.pop(j)
        (cluster2,k2) = lista_clusters[i]
        matriz_enlace.append([k1,k2,d_minima,cluster1.num_puntos() + cluster2.num_puntos()])
        new_cluster = cluster1.combinar(cluster2)
        lista_clusters[i] = (new_cluster, n + s)
        if d_minima <= maximo:
            resultado.pop(j)
            resultado[i] = new_cluster
        for l in range(i):
            M[l,i] = lance_williams(M,l,i,j,a,b,c,d)
        M = np.delete(np.delete(M, j, axis = 0),j,axis = 1)
    return (matriz_enlace, resultado)
        

        
        

        
        
    
    
    
    
    
    
    
    
    
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
    
    
    
    
    
    
    
    
    
    