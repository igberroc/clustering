# -*- coding: utf-8 -*-

import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from points import Point
from fuzzy import fuzzy_cmeans

def decompose_x_y(lista: list[Point]) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for point in lista:
       x.append(point.coordinates[0])
       y.append(point.coordinates[1])
    return x,y


def main1():
    delta = 0.25
    npoints = 20
    data1 = [Point(1 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data2 = [Point(1 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data3 = [Point(0 + random.uniform(-delta,delta), 1 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data4 = [Point(0 + random.uniform(-delta,delta), 0 + random.uniform(-delta, delta )) for _ in range(npoints)]
    data = data1 + data2 + data3 + data4
    (x,y) = decompose_x_y(data)

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
    (x, y) = decompose_x_y(dataset)

    initial_centroids = [Point(random.uniform(-5,5),random.uniform(-5,5)) for _ in range(3)]
    membership_matrix = fuzzy_cmeans(dataset, initial_centroids,2,3,0.001, 100)
    for i in range(3):
        plt.figure()
        scatter = plt.scatter(x,y, c = membership_matrix[i,:], cmap = plt.cm.Blues, s = 50)
        plt.colorbar(scatter, label=f'Membership degree to cluster {i+1}')
        plt.title(f'Cluster {i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


def main_iris():
    iris = load_iris()
    data = iris.data[2:,:4]
    dataset = [Point(coord[0], coord[1]) for coord in data]
    (x, y) = decompose_x_y(dataset)

    initial_centroids = [Point(random.uniform(min(x), max(x)), random.uniform(min(y), max(y))) for _ in range(3)]
    membership_matrix = fuzzy_cmeans(dataset, initial_centroids, 2, 3, 0.001, 100)

    for i in range(3):
        plt.figure()
        scatter = plt.scatter(x, y, c = membership_matrix[i, :], cmap = plt.cm.Blues, s = 50)
        plt.colorbar(scatter, label=f'Membership degree to cluster {i+1}')
        plt.title(f'Cluster {i+1}')
        plt.xlabel('Sepal length (cm)')
        plt.ylabel('Sepal width (cm)')
        plt.show()
    
