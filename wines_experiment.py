
import sys
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import random
import numpy as np

from points import Point
from agglomerative import median, complete, average, ward
from experiment_functions import kmeans_exp, agglomerative_exp, fuzzy_exp, dbscan_exp, em_exp, table_plot


def main_raw():
    """
    Experiment with wines dataset. It saves two files: one with the table results, and other
    with the dendrograms of agglomerative clustering.
    In all algorithms, 3 clusters were sought.
    """
    df = pd.read_csv('wine_dataset.csv')
    df_tuples = df.itertuples(index = False, name = None)
    data = []
    for point_coordinates in df_tuples:        #Changing data into points.
        data.append(Point(*point_coordinates))
    results = []

    kmeans_results = kmeans_exp(data, 3, 0.001, 100)
    results.append(kmeans_results)

    agglomerative_results1, linkage_matrix1 = agglomerative_exp(data, complete, 550)
    results.append(agglomerative_results1)

    agglomerative_results2, linkage_matrix2 = agglomerative_exp(data, median, 300)
    results.append(agglomerative_results2)

    initial_centroids = []
    for _ in range(3):              #One centroid for each cluster.
        point_coordinates = tuple([random.uniform(-2, 2) for _ in range(13)])
        initial_centroids.append(Point(*point_coordinates))
    fuzzy_results = fuzzy_exp(data, initial_centroids, 2, 3, 0.001, 100)
    results.append(fuzzy_results)

    test_parameters = [(40,7), (50,13), (47,8)]
    for (eps, min_points) in test_parameters:
        dbscan_results = dbscan_exp(data, eps, min_points)
        results.append(dbscan_results)
    """
    initial_covariances = np.diag([1,1,0.1,10,100,1,1,0.1,1,1,0.1,1,10000])  (variance estimations)
    """
    initial_covariances = [12_000 * np.eye(13) for _ in range(3)]
    em_results = em_exp(data, 3, initial_covariances, 1e-20, 100)
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
    plt.savefig("dendrograms_wines_raw.svg", format = "svg")

    table_plot(results, "Wine clustering", "results_wines_raw.svg")


def main_standarized():
    """
    Experiment with wines dataset, but standardizing the data. It saves two files: one with the table results,
    and other with the dendrograms of agglomerative clustering.
    In all algorithms, 3 clusters were sought.
    """
    df = pd.read_csv('wine_dataset.csv')
    scaler = StandardScaler()
    features = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids',
                'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']
    df[features] = scaler.fit_transform(df[features])
    df_tuples = df.itertuples(index = False, name = None)
    data = []
    for point_coordinates in df_tuples:    #Changing data into points.
        data.append(Point(*point_coordinates))
    results = []

    kmeans_results = kmeans_exp(data, 3, 0.001, 100)
    results.append(kmeans_results)

    agglomerative_results1, linkage_matrix1 = agglomerative_exp(data, average, 6.5)
    results.append(agglomerative_results1)

    agglomerative_results2, linkage_matrix2 = agglomerative_exp(data, ward, 20)
    results.append(agglomerative_results2)

    initial_centroids = []
    for i in range(3):             #One centroid for each cluster.
        point_coordinates = tuple([random.uniform(-2, 2) for _ in range(13)])
        initial_centroids.append(Point(*point_coordinates))
    fuzzy_results = fuzzy_exp(data, initial_centroids, 1.25, 3, 0.001, 100)
    results.append(fuzzy_results)

    test_parameters = [(2.55, 22), (2.25,13), (2.15,10)]
    for (eps, min_points) in test_parameters:
        dbscan_results = dbscan_exp(data, eps, min_points)
        results.append(dbscan_results)

    initial_covariances = [np.eye(13) for _ in range(3)]
    em_results = em_exp(data, 3, initial_covariances, 1e-20, 100)
    results.append(em_results)

    #Plots
    fig, axes = plt.subplots(1, 2, figsize = (20, 4))
    dendrogram(linkage_matrix1, ax = axes[0], leaf_rotation = 90, leaf_font_size = 3)
    axes[0].set_title("Average linkage")
    axes[0].set_xlabel("cluster indexes")
    axes[0].set_ylabel("distance between clusters")
    dendrogram(linkage_matrix2, ax=axes[1], leaf_rotation = 90, leaf_font_size = 3)
    axes[1].set_title("Ward linkage")
    axes[1].set_xlabel("cluster indexes")
    axes[1].set_ylabel("distance between clusters")
    plt.savefig("dendrograms_wines_stand.svg", format = "svg")


    table_plot(results, "Wine clustering (standardized data)", "results_wines_stand.svg")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiment_number>")
        exit(1)
    number = int(sys.argv[1])
    if number == 1:
        main_raw()
    elif number == 2:
        main_standarized()
    else:
        print("Invalid experiment number")





