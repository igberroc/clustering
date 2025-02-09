
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
from agglomerative import agglomerative
from fuzzy import fuzzy_cmeans
from dbscan import dbscan
from EM import em

if __name__ == '__main__':
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
    results.append(['KMeans', silhouette, db, c, t1 - t0])


    #AGGLOMERATIVE
    t0 = perf_counter()
    (linkage_matrix, list_clusters) = agglomerative(data, 0.5, 0.5, 0, 0.5, 550)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['Agglomerative(average linkage, max_dist = 100) ', silhouette, db, c, t1 - t0])


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
    membership_matrix = fuzzy_cmeans(data, initial_centroids, 2, 3, 0.001, 100)
    t1 = perf_counter()
    list_clusters = [Cluster() for _ in range(3)]
    for i in range(len(data)):
        max_index = np.argmax(membership_matrix[:,i])
        list_clusters[max_index].add_point(data[i])
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['Fuzzy', silhouette, db, c, t1 - t0])

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
        print(len(list_clusters))
        print(noise.size())
    print(results)

    #EM
    """
    
    scaler = StandardScaler()
    features = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids',
                'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']
    df[features] = scaler.fit_transform(df[features])
    """
    t0 = perf_counter()
    list_clusters = em(data,3,0.001,100)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    results.append(['EM', silhouette, db, c, t1 - t0])
    print(results)









