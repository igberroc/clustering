
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd

from points import Point
from kmeans import kmeans
from metrics import silhouette_index, db_index, c_index
from agglomerative import agglomerative

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
    results.append(['Agglomerative(complete linkage, max_dist = 550) ', silhouette, db, c, t1 - t0])
    print(results)
    print(len(list_clusters))

    plt.figure()
    dendrogram(linkage_matrix)
    plt.xlabel("clusters indexes")
    plt.ylabel("distance between clusters")
    plt.show()






