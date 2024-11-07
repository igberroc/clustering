# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:42:14 2024

@author: Nacho
"""
import numpy as np


from typing import List
from puntos import Distancia, Clusters, Lance_Williams


def matriz_proximidad(lista_clusters:List, a:float, b:float, c:float, d:float, metodo = 'Distancia') -> np.ndarray:
    n = len(lista_clusters)
    LW = Lance_Williams()
    matriz_proximidad = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i+1,n):
            cluster1 = lista_clusters[i][0]
            cluster2 = lista_clusters[j][0]
            dist = LW.d_clusters(cluster1, cluster2, a, b, c, d, metodo)
            matriz_proximidad[i, j] = dist
    return matriz_proximidad

def minimo(m):
    n = len(m)
    minimo = m[0,1]
    i_minimo = 0
    j_minimo = 1
    for i in range(n-1):
        for j in range(i+1,n):
            if m[i,j] < minimo:
                minimo = m[i,j]
                i_minimo = i
                j_minimo = j
    return (minimo,(i_minimo,j_minimo))
            

def aglomerativo(datos:list, a:float, b:float, c:float, d:float, metodo:'Distancia'):
    n = len(datos)
    matriz_enlace = []
    lista_clusters = []
    for k in range(n):
        cluster = Clusters()
        cluster.agregar_punto(datos[k])
        lista_clusters.append((cluster,k))
    for m in range(n-1):
        M = matriz_proximidad(lista_clusters,a,b,c,d,metodo)
        (d_minima,(i,j)) = minimo(M)
        indices_ordenados = sorted([i,j])
        (cluster1,k1) = lista_clusters.pop(indices_ordenados[1])
        (cluster2,k2) = lista_clusters.pop(indices_ordenados[0])
        matriz_enlace.append([k1,k2,d_minima,cluster1.num_puntos() + cluster2.num_puntos()])
        new_cluster = cluster1.combinar(cluster2)
        lista_clusters.append((new_cluster, n + m))
    return matriz_enlace
        

        
        

        
        
    
    
    
    
    
    
    
    
    
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
    
    
    
    
    
    
    
    
    
    