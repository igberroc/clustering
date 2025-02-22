
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import random
import numpy as np

from points import Point
from agglomerative import median, complete, average
from test_functions import kmeans_test, agglomerative_test, fuzzy_test, dbscan_test, em_test, table_plot


def main1():
    df = pd.read_csv('wine_dataset.csv')
    df_tuples = df.itertuples(index = False, name = None)
    data = []
    for point_coordinates in df_tuples:
        data.append(Point(*point_coordinates))
    results = []

    kmeans_results = kmeans_test(data, 3, 0.001, 100)
    results.append(kmeans_results)

    agglomerative_results1, linkage_matrix1 = agglomerative_test(data, complete, 550)
    results.append(agglomerative_results1)

    agglomerative_results2, linkage_matrix2 = agglomerative_test(data, median, 300)
    results.append(agglomerative_results2)

    initial_centroids = []
    for _ in range(3):
        point_coordinates = tuple([random.uniform(-2, 2) for _ in range(13)])
        initial_centroids.append(Point(*point_coordinates))
    fuzzy_results = fuzzy_test(data, initial_centroids, 2, 3, 0.001, 100)
    results.append(fuzzy_results)

    test_parameters = [(40,7), (50,13), (47,8)]
    for (eps, min_points) in test_parameters:
        dbscan_results = dbscan_test(data, eps, min_points)
        results.append(dbscan_results)
    """
    initial_covariances = np.diag([1,1,1,10,100,1,1,1,1,1,1,1,10000])   (variance estimations)
    """
    initial_covariances = [12_000 * np.eye(13) for _ in range(3)]
    em_results = em_test(data, 3, initial_covariances, 1e-20, 100)
    results.append(em_results)

    #Plots
    fig, axes = plt.subplots(1, 2, figsize = (20, 4))
    dendrogram(linkage_matrix1, ax = axes[0], leaf_rotation = 90, leaf_font_size = 3)
    axes[0].set_title("Complete linkage")
    axes[0].set_xlabel("cluster indexes")
    axes[0].set_ylabel("distance between clusters")
    dendrogram(linkage_matrix2, ax=axes[1], leaf_rotation = 90, leaf_font_size = 3)
    axes[1].set_title("Median linkage")
    axes[1].set_xlabel("cluster indexes")
    axes[1].set_ylabel("distance between clusters")
    plt.savefig("dendrogram1.svg", format = "svg")

    table_plot(results, "Wine clustering", "results1.svg")

#Standarized data
def main2():
    df = pd.read_csv('wine_dataset.csv')
    scaler = StandardScaler()
    features = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids',
                'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']
    df[features] = scaler.fit_transform(df[features])
    df_tuples = df.itertuples(index = False, name = None)
    data = []
    for point_coordinates in df_tuples:
        data.append(Point(*point_coordinates))
    results = []

    kmeans_results = kmeans_test(data, 3, 0.001, 100)
    results.append(kmeans_results)

    agglomerative_results, linkage_matrix = agglomerative_test(data, average, 6.5)
    results.append(agglomerative_results)

    initial_centroids = [0 for _ in range(3)]
    for i in range(3):
        point_coordinates = tuple([random.uniform(-2, 2) for _ in range(13)])
        initial_centroids[i] = Point(*point_coordinates)
    fuzzy_results = fuzzy_test(data, initial_centroids, 1.25, 3, 0.001, 100)
    results.append(fuzzy_results)

    test_parameters = [(3,2), (2.5,2), (2.8,2)]
    for (eps, min_points) in test_parameters:
        dbscan_results = dbscan_test(data, eps, min_points)
        results.append(dbscan_results)

    initial_covariances = [np.eye(13) for _ in range(3)]
    em_results = em_test(data, 3, initial_covariances, 1e-20, 100)
    results.append(em_results)

    #Plots
    plt.figure(figsize = (10, 4))
    dendrogram(linkage_matrix, leaf_rotation = 90, leaf_font_size = 3)
    plt.xlabel("clusters indexes")
    plt.ylabel("distance between clusters")
    plt.savefig("dendrogram2.svg", format = "svg")

    table_plot(results, "Wine clustering (standardized data)", "results2.svg")

if __name__ == "__main__":
    print("Choose experiment: 1 or 2")
    number = input()
    while number not in ["1", "2"]:
        print("Choose a correct number: 1 or 2")
        number = input()

    if number == "1":
        main1()
    elif number == "2":
        main2()





