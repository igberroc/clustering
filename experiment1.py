
from sklearn.preprocessing import StandardScaler
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import random
import numpy as np

from points import Point, Cluster
from kmeans import kmeans
from metrics import silhouette_index, db_index, c_index
from agglomerative import agglomerative, complete, average
from fuzzy import fuzzy_cmeans
from dbscan import dbscan
from EM import em



def main1():
    df = pd.read_csv('wine_dataset.csv')
    df_tuples = df.itertuples(index = False, name = None)
    data = []
    for point_coordinates in df_tuples:
        data.append(Point(*point_coordinates))

    results = []

    #KMEANS
    t0 = perf_counter()
    list_clusters = kmeans(data,3,0.001,100)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['KMeans(eps = 0.001, max_iter = 100)', silhouette, db, c, t1 - t0])

    #AGGLOMERATIVE
    t0 = perf_counter()
    (linkage_matrix, list_clusters) = agglomerative(data, complete, 550)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['Agglomerative(complete linkage, max_dist = 550) ', silhouette, db, c, t1 - t0])

    plt.figure(figsize = (10, 4))
    dendrogram(linkage_matrix, leaf_rotation = 90, leaf_font_size = 3)
    plt.xlabel("clusters indexes")
    plt.ylabel("distance between clusters")
    plt.savefig("dendrogram1.svg", format = "svg")


    #FUZZY
    initial_centroids = [0 for _ in range(3)]
    for i in range(3):
        point_coordinates = tuple([random.uniform(-2, 2) for _ in range(13)])
        initial_centroids[i] = Point(*point_coordinates)
    t0 = perf_counter()
    membership_matrix = fuzzy_cmeans(data, initial_centroids, 2, 3, 0.001, 100)
    t1 = perf_counter()
    list_clusters = [Cluster() for _ in range(3)]
    for i in range(len(data)):
        max_index = np.argmax(membership_matrix[:,i])
        list_clusters[max_index].add_point(data[i])
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['Fuzzy(m = 2, eps = 0.001, max_iter = 100)', silhouette, db, c, t1 - t0])

    #DBSCAN
    test_parameters = [(40,7), (50,13), (47,8)]
    for (eps, min_points) in test_parameters:
        t0 = perf_counter()
        (list_clusters, noise) = dbscan(data, eps, min_points)
        t1 = perf_counter()
        silhouette = silhouette_index(list_clusters)
        db = db_index(list_clusters)
        c = c_index(data, list_clusters)
        results.append([f'DBSCAN (eps = {eps}, min_points = {min_points})', silhouette, db, c, t1 - t0])

    #EM
    """
    array = np.array([list(point.get_coordinates()) for point in data])
    media = np.mean(array, axis = 0)
    print(media)
    varianza = np.var(array, axis = 0, ddof = 0)
    print(varianza)
    """
    cov_matrix = np.diag([1,1,1,10,100,1,1,1,1,1,1,1,10000])
    initial_covariances = [12_000*np.eye(13) for _ in range(3)]
    t0 = perf_counter()
    list_clusters = em(data,3, initial_covariances,1e-20,100)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['EM', silhouette, db, c, t1 - t0])

    #Results
    df_results = pd.DataFrame(results,
                              columns=["Algorithm", "Silhouette index", "Davies-Bouldin index","C-index", "Time(s)"])
    plt.figure(figsize=(10, 4))
    plt.title("Wine clustering", fontsize=14, fontweight='bold')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    table = plt.table(cellText=df_results.round(3).values,
                      colLabels=df_results.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2, 3, 4])
    plt.savefig("results1.svg", format = "svg")

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

    #KMEANS
    t0 = perf_counter()
    list_clusters = kmeans(data,3,0.001,100)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['KMeans(eps = 0.001, max_iter = 100)', silhouette, db, c, t1 - t0])

    #AGGLOMERATIVE
    t0 = perf_counter()
    (linkage_matrix, list_clusters) = agglomerative(data, average, 6.5)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['Agglomerative(average linkage, max_dist = 6.5) ', silhouette, db, c, t1 - t0])

    plt.figure()
    dendrogram(linkage_matrix)
    plt.xlabel("clusters indexes")
    plt.ylabel("distance between clusters")
    plt.show()

    #FUZZY
    initial_centroids = [0 for _ in range(3)]
    for i in range(3):
        point_coordinates = tuple([random.uniform(-2, 2) for _ in range(13)])
        initial_centroids[i] = Point(*point_coordinates)
    t0 = perf_counter()
    membership_matrix = fuzzy_cmeans(data, initial_centroids, 1.25, 3, 0.001, 100)
    t1 = perf_counter()
    list_clusters = [Cluster() for _ in range(3)]
    for i in range(len(data)):
        max_index = np.argmax(membership_matrix[:,i])
        list_clusters[max_index].add_point(data[i])
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['Fuzzy(m = 1.25, eps = 0.001, max_iter = 100)', silhouette, db, c, t1 - t0])


    #DBSCAN
    test_parameters = [(3,2), (2.5,2), (2.8,2)]
    for (eps, min_points) in test_parameters:
        t0 = perf_counter()
        (list_clusters, noise) = dbscan(data, eps, min_points)
        t1 = perf_counter()
        silhouette = silhouette_index(list_clusters)
        db = db_index(list_clusters)
        c = c_index(data, list_clusters)
        results.append([f'DBSCAN (eps = {eps}, min_points = {min_points})', silhouette, db, c, t1 - t0])

    #EM
    initial_covariances = [np.eye(13) for _ in range(3)]
    t0 = perf_counter()
    list_clusters = em(data,3, initial_covariances,1e-20,100)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['EM', silhouette, db, c, t1 - t0])

    #Results
    df_results = pd.DataFrame(results,
                              columns=["Algorithm", "Silhouette index", "Davies-Bouldin index","C-index", "Time(s)"])
    plt.figure(figsize=(10, 4))
    plt.title("Wine clustering (standardized data)", fontsize=14, fontweight='bold')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    table = plt.table(cellText=df_results.round(3).values,
                      colLabels=df_results.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2, 3, 4])
    plt.show()

main1()



