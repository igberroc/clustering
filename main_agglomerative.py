# -*- coding: utf-8 -*-


import random

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from agglomerative import agglomerative, ward, average
from points import Point, decompose_x_y
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

    (x1, y1) = decompose_x_y(result[0])
    (x2, y2) = decompose_x_y(result[1])
    (x3, y3) = decompose_x_y(result[2])
    (x4, y4) = decompose_x_y(result[3])

    plt.figure()
    plt.plot(x1, y1, 'o', markerfacecolor='red')
    plt.plot(x2, y2, 'o', markerfacecolor='blue')
    plt.plot(x3, y3, 'o', markerfacecolor='green')
    plt.plot(x4, y4, 'o', markerfacecolor='yellow')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters agglomerative")
    plt.legend()
    plt.show()

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

    
main1()
