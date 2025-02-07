# -*- coding: utf-8 -*-

import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
import numpy as np

from points import Point, Cluster, descompose_x_y
from dbscan import dbscan
from metrics import silhouette_index, db_index, c_index



def random_color():
    return f'#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}'


def main1():
    df = pd.read_csv('C:/Users/nacho/Downloads/dataset_dbscan.csv')
    list_x = df['Weight'].tolist()
    list_y = df['Height'].tolist()
    data = []
    for i,j in zip(list_x,list_y):
        data.append(Point(i,j))
        
    eps = 0.6
    min_points = 5
    (list_clusters, noise) = dbscan(data, eps, min_points)
    
    used_colors = set()
    for cluster in list_clusters:
        (x,y) = descompose_x_y(cluster)
        color = random_color()
        while color in used_colors:
            color = random_color()
        used_colors.add(color)
        plt.scatter(x,y, s = 10, color = color)
    
    silhouette = silhouette_index(list_clusters)
    print(f"The Silhouette index is: {silhouette} ")
    
    db = db_index(list_clusters)
    print(f"The Davies-Bouldin index is: {db} ")

    c = c_index(data, list_clusters)
    print(f"The C-index is: {c} ")

    (x,y) = descompose_x_y(noise)
    plt.scatter(x,y, s = 10, color = 'black')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters DBSCAN")
    plt.legend()
    plt.show()
    
def main2():
    dataset, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    data = []
    for [i,j] in dataset:
        data.append(Point(i,j))
    eps = 0.2
    min_points = 5
    (list_clusters, noise) = dbscan(data, eps, min_points)
    
    used_colors = set()
    for cluster in list_clusters:
        (x,y) = descompose_x_y(cluster)
        color = random_color()
        while color in used_colors:
            color = random_color()
        used_colors.add(color)
        plt.scatter(x,y, s = 10, color = color)
    
    silhouette = silhouette_index(list_clusters)
    print(f"The Silhouette index is: {silhouette} ")
    
    db = db_index(list_clusters)
    print(f"The Davies-Bouldin index is: {db} ")

    c = c_index(data, list_clusters)
    print(f"The C-index is: {c} ")

    (x,y) = descompose_x_y(noise)
    plt.plot(x,y, 'o', color = 'black')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters DBSCAN")
    plt.legend()
    plt.show()
    

def main3():

    def twospirals(n_points, noise=.5):
        epsilon = 0.1
        n = (np.random.rand(n_points,1)+epsilon) * 780 * (2*np.pi)/360
        d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
        d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
        c_1 = np.hstack((d1x,d1y))
        c_2 = np.hstack((-d1x,-d1y))
        return np.vstack((c_1, c_2))
    
    n_points = 500
    dataset = twospirals(n_points)
    data = []
    for [x1,x2] in dataset:
        data.append(Point(x1,x2))
    eps = 1.7
    min_points = 2
    (list_clusters, noise) = dbscan(data, eps, min_points)
    
    used_colors = set()
    for cluster in list_clusters:
        (x,y) = descompose_x_y(cluster)
        color = random_color()
        while color in used_colors:
            color = random_color()
        used_colors.add(color)
        plt.scatter(x,y, s = 10, color = color)
    
    silhouette = silhouette_index(list_clusters)
    print(f"The Silhouette index is: {silhouette} ")
    
    db = db_index(list_clusters)
    print(f"The Davies-Bouldin index is: {db} ")

    c = c_index(data, list_clusters)
    print(f"The C-index is: {c} ")

    (x,y) = descompose_x_y(noise)
    plt.plot(x,y, 'o', color = 'black')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters DBSCAN")
    plt.legend()
    plt.show()
        
        
   



    



