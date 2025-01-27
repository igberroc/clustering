# -*- coding: utf-8 -*-

import seaborn as sns
import random
import matplotlib.pyplot as plt

from puntos import Point
from fuzzy import fuzzy_cmeans

def descompose_x_y(lista: list[Point]) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for Point in lista:
       x.append(Point.coordinates[0])
       y.append(Point.coordinates[1])
    return (x,y)


def main1():
    delta = 0.25
    npoints = 20
    data1 = [Point(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data2 = [Point(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data3 = [Point(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data4 = [Point(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data = data1 + data2 + data3 + data4
    (x,y) = descompose_x_y(data)

    initial_centroids = [Point(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(4)]
    membership_matrix = fuzzy_cmeans(data, initial_centroids,2,4,0.001, 100)

    for i in range(4):
        plt.figure()
        scatter = plt.scatter(x,y, c = membership_matrix[i,:], cmap=plt.cm.Blues, s = 50)
        plt.colorbar(scatter, label=f'Membership degree to cluster {i+1}')
        plt.title(f'Cluster {i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    
def main_penguins():
    penguins = sns.load_dataset('penguins').dropna()
    dataset = penguins[['bill_length_mm', 'bill_depth_mm']].values.tolist()
    for i in range(len(dataset)):
        dataset[i] = Point(dataset[i][0],dataset[i][1])
    (x, y) = descompose_x_y(dataset)

    initial_centroids = [Point(random.uniform(-5,5),random.uniform(-5,5)) for _ in range(3)]
    membership_matrix = fuzzy_cmeans(dataset, initial_centroids,2,3,0.001, 100)
    for i in range(3):
        plt.figure()
        scatter = plt.scatter(x,y, c = membership_matrix[i,:], cmap=plt.cm.Blues, s = 50)
        plt.colorbar(scatter, label=f'Membership degree to cluster {i+1}')
        plt.title(f'Cluster {i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    

