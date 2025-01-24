# -*- coding: utf-8 -*-


from points import Distance, Distance_euclidea, Point, Cluster
import math


#Minimo entre la Distance promedio de un Point a otro cluster al cual no pertenece.
def dist_minima_Point_cluster(Point: Point, lista_cluster: list[Cluster],
                              Distance: Distance = Distance_euclidea) -> float:
    minimo = math.inf
    b = 0
    for cluster in lista_cluster:
        for Point_cluster in cluster.points:
            b += Distance(Point,Point_cluster)
        b = b / cluster.size()
        minimo = min(b,minimo)
        b = 0
    return minimo

#Distance promedio de un Point a su propio cluster
def dist_propio_cluster(Point: Point, cluster: Cluster, Distance: Distance = Distance_euclidea) -> float:
    a = 0
    for Point_cluster in cluster.points:
        a += Distance(Point,Point_cluster)
    if cluster.size() != 1:
        a = a / (cluster.size()-1)
    return a
            


#Indice de Silueta de un Point dado.
def indice_Silueta_Point(Point: Point, propio_cluster: Cluster, 
                         resto_clusters: list[Cluster], Distance: Distance = Distance_euclidea ) -> float:
    b = dist_minima_Point_cluster(Point,resto_clusters, Distance)
    a = dist_propio_cluster(Point, propio_cluster, Distance)
    indice = (b - a) / max(a,b)
    return indice

# -1 indica mala clasficacion, 1 indica buena clasificacion
def indice_Silueta(lista_cluster: list[Cluster], Distance: Distance = Distance_euclidea) -> float:
    num_points = 0
    indice = 0
    for _ in range(len(lista_cluster)):
        cluster = lista_cluster.pop(0)
        for Point in cluster.points:
            indice += indice_Silueta_Point(Point,cluster,lista_cluster, Distance)
        num_points += cluster.size()
        lista_cluster.append(cluster)
    return (indice / num_points)
        
    
def dispersion_cluster(cluster: Cluster, Distance: Distance = Distance_euclidea) -> (float,float):
    centroid = cluster.centroid()
    dispersion = 0
    for Point in cluster.points:
        dispersion += Distance(Point,centroid)
    return (dispersion / cluster.size(),centroid)


#Indice de Davies-Bouldin, entre 0 e infinito, cuanto mas bajo, mejor clasificacion
def indice_DB(lista_cluster: list[Cluster], Distance: Distance = Distance_euclidea) -> float:
    k = len(lista_cluster)
    lista_dispersion = []
    lista_centroid = []
    indice = 0
    for cluster in lista_cluster:
        (dispersion, centroid) = dispersion_cluster(cluster,Distance)
        lista_dispersion.append(dispersion)
        lista_centroid.append(centroid)
    for i in range(k):
        maximo = 0
        for j in range(k):
            if i != j:
                expresion = (lista_dispersion[i] + lista_dispersion[j])/Distance(lista_centroid[i],lista_centroid[j])
                maximo = max(maximo, expresion)
        indice += maximo
    return (indice / k)
                
                
    
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    