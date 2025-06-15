# -*- coding: utf-8 -*-



import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_moons

from kmeans import kmeans
from points import Point, decompose_x_y, gower_distance
from metrics import silhouette_index, db_index, c_index, ch_index, dunn_index


def main1():
    delta = 0.25
    n_points = 100
    data1 = [Point(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(n_points)]
    data2 = [Point(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(n_points)]
    data3 = [Point(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(n_points)]
    data4 = [Point(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(n_points)]
    data = data1 + data2 + data3 + data4
    
    list_clusters = kmeans(data,4,0.001,100)
    silhouette = silhouette_index(list_clusters)
    print(f"The Silhouette index is: {silhouette} ")
    
    db = db_index(list_clusters)
    print(f"The Davies-Bouldin index is: {db} ")

    c = c_index(data, list_clusters)
    print(f"The C-index is: {c} ")

    ch = ch_index(list_clusters)
    print(f"The Calinski-Harabasz index is: {ch} ")

    dunn = dunn_index(list_clusters)
    print(f"The Dunn index is: {dunn} ")

    (x1,y1) = decompose_x_y(list_clusters[0])
    (x2,y2) = decompose_x_y(list_clusters[1])
    (x3,y3) = decompose_x_y(list_clusters[2])
    (x4,y4) = decompose_x_y(list_clusters[3])  
    
    plt.plot(x1,y1,'o',markerfacecolor = 'red') 
    plt.plot(x2,y2,'o',markerfacecolor = 'blue') 
    plt.plot(x3,y3,'o',markerfacecolor = 'green')
    plt.plot(x4,y4,'o',markerfacecolor = 'yellow')
    
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters k-means")
    plt.legend()
    plt.show()
    

def main2():
    dataset, y = make_moons(n_samples = 300, noise = 0.1, random_state = 42)
    data = []
    for [i,j] in dataset:
        data.append(Point(i,j))
    
    list_clusters = kmeans(data,2,0.001,100)
    silhouette = silhouette_index(list_clusters)
    print(f"The Silhouette index is: {silhouette} ")
    
    db = db_index(list_clusters)
    print(f"The Davies-Bouldin index is: {db} ")

    c = c_index(data, list_clusters)
    print(f"The C-index is: {c} ")

    ch = ch_index(list_clusters)
    print(f"The Calinski-Harabasz index is: {ch} ")

    dunn = dunn_index(list_clusters)
    print(f"The Dunn index is: {dunn} ")

    (x1,y1) = decompose_x_y(list_clusters[0])
    (x2,y2) = decompose_x_y(list_clusters[1])
    
    plt.plot(x1,y1,'o',markerfacecolor = 'red') 
    plt.plot(x2,y2,'o',markerfacecolor = 'blue') 
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters k-means")
    plt.legend()
    plt.show()






    
    
            