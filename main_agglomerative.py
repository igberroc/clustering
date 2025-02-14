# -*- coding: utf-8 -*-


import random

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from agglomerative import agglomerative, single, complete, ward, average
from points import Point
from metrics import silhouette_index, db_index

def main1():
    delta = 0.25
    npoints = 20
    data1 = [Point(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data2 = [Point(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data3 = [Point(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data4 = [Point(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data = data1 + data2 + data3 + data4
    
    (linkage_matrix,result) = agglomerative(data, average, 0.6)
    silhouette = silhouette_index(result)
    print(f"The Silhouette index is: {silhouette} ")
    
    
    db = db_index(result)
    print(f"The Davies-Bouldin index is: {db} ")
    
    plt.figure()
    dendrogram(linkage_matrix)
    plt.xlabel("clusters indexes")
    plt.ylabel("distance between clusters")
    plt.show()


def main2():
    iris = load_iris()
    iris_list = iris.data.tolist()
    data = []
    for elem in iris_list:
        p = Point()
        p.coordinates = tuple(elem)
        data.append(p)
    (linkage_matrix,result) = agglomerative(data,ward,10)
    print(len(result))
    
    silhouette = silhouette_index(result)
    print(f"The silhouette index is: {silhouette} ")
    db = db_index(result)
    print(f"The Davies-Bouldin index is: {db} ")
    
    
    plt.figure()
    dendrogram(linkage_matrix)
    plt.xlabel("clusters indexes")
    plt.ylabel("distance between clusters")
    plt.show()

    

