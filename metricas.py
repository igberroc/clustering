# -*- coding: utf-8 -*-


from puntos import Distancia, distancia_euclidea, Punto, Cluster
import math


#Minimo entre la distancia promedio de un punto a otro cluster al cual no pertenece.
def dist_minima_punto_cluster(punto: Punto, lista_cluster: list[Cluster],
                              distancia: Distancia = distancia_euclidea) -> float:
    minimo = math.inf
    b = 0
    for cluster in lista_cluster:
        for punto_cluster in cluster.puntos:
            b += distancia(punto,punto_cluster)
        b = b / cluster.tamaño()
        minimo = min(b,minimo)
        b = 0
    return minimo

#Distancia promedio de un punto a su propio cluster
def dist_propio_cluster(punto: Punto, cluster: Cluster, distancia: Distancia = distancia_euclidea) -> float:
    a = 0
    for punto_cluster in cluster.puntos:
        a += distancia(punto,punto_cluster)
    if cluster.tamaño() != 1:
        a = a / (cluster.tamaño()-1)
    return a
            


#Indice de Silueta de un punto dado.
def indice_Silueta_punto(punto: Punto, propio_cluster: Cluster, 
                         resto_clusters: list[Cluster], distancia: Distancia = distancia_euclidea ) -> float:
    b = dist_minima_punto_cluster(punto,resto_clusters, distancia)
    a = dist_propio_cluster(punto, propio_cluster, distancia)
    indice = (b - a) / max(a,b)
    return indice

# -1 indica mala clasficacion, 1 indica buena clasificacion
def indice_Silueta(lista_cluster: list[Cluster], distancia: Distancia = distancia_euclidea) -> float:
    num_puntos = 0
    indice = 0
    for _ in range(len(lista_cluster)):
        cluster = lista_cluster.pop(0)
        for punto in cluster.puntos:
            indice += indice_Silueta_punto(punto,cluster,lista_cluster, distancia)
        num_puntos += cluster.tamaño()
        lista_cluster.append(cluster)
    return (indice / num_puntos)
        
    
def dispersion_cluster(cluster: Cluster, distancia: Distancia = distancia_euclidea) -> (float,float):
    centroide = cluster.calcular_centroide()
    dispersion = 0
    for punto in cluster.puntos:
        dispersion += distancia(punto,centroide)
    return (dispersion / cluster.tamaño(),centroide)


#Indice de Davies-Bouldin, entre 0 e infinito, cuanto mas bajo, mejor clasificacion
def indice_DB(lista_cluster: list[Cluster], distancia: Distancia = distancia_euclidea) -> float:
    k = len(lista_cluster)
    lista_dispersion = []
    lista_centroide = []
    indice = 0
    for cluster in lista_cluster:
        (dispersion, centroide) = dispersion_cluster(cluster,distancia)
        lista_dispersion.append(dispersion)
        lista_centroide.append(centroide)
    for i in range(k):
        maximo = 0
        for j in range(k):
            if i != j:
                expresion = (lista_dispersion[i] + lista_dispersion[j])/distancia(lista_centroide[i],lista_centroide[j])
                maximo = max(maximo, expresion)
        indice += maximo
    return (indice / k)
                
                
    
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    