
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from points import Cluster, euclidean_distance
from kmeans import kmeans
from metrics import silhouette_index, db_index, c_index
from agglomerative import agglomerative
from fuzzy import fuzzy_cmeans
from dbscan import dbscan
from EM import em


def kmeans_test(data, k, eps, max_iter, dist = euclidean_distance):
    t0 = perf_counter()
    list_clusters = kmeans(data, k, eps, max_iter, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    return [f'KMeans(eps = {eps}, max_iter = {max_iter})', silhouette, db, c, t1 - t0]

def agglomerative_test(data, method, max_dist = 0, dist = euclidean_distance):
    t0 = perf_counter()
    (linkage_matrix, list_clusters) = agglomerative(data, method, max_dist, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    return [f'Agglomerative({method.__name__} linkage, max_dist = {max_dist}) ', silhouette, db, c, t1 - t0], linkage_matrix

def fuzzy_test(data, initial_centroids, m, c, eps, max_iter, dist = euclidean_distance):
    t0 = perf_counter()
    membership_matrix = fuzzy_cmeans(data, initial_centroids, m, c, eps, max_iter, dist)
    t1 = perf_counter()
    list_clusters = [Cluster() for _ in range(c)]
    for i in range(len(data)):
        max_index = np.argmax(membership_matrix[:,i])
        list_clusters[max_index].add_point(data[i])
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    return [f'Fuzzy(m = {m}, eps = {eps}, max_iter = {max_iter})', silhouette, db, c, t1 - t0]

def dbscan_test(data, eps, min_points, dist = euclidean_distance):
    t0 = perf_counter()
    (list_clusters, noise) = dbscan(data, eps, min_points, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    return [f'DBSCAN(eps = {eps}, min_points = {min_points})', silhouette, db, c, t1 - t0]

def em_test(data, n_clusters, initial_covariances, eps, max_iter):
    t0 = perf_counter()
    list_clusters = em(data,n_clusters, initial_covariances, eps ,max_iter)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    return [f'EM(eps = {eps}, max_iter = {max_iter}', silhouette, db, c, t1 - t0]

def table_plot(results: list[list], plot_title: str, filename: str):
    df_results = pd.DataFrame(results,
                              columns=["Algorithm", "Silhouette index", "Davies-Bouldin index", "C-index", "Time(s)"])
    plt.figure(figsize=(10, 4))
    plt.title(plot_title, fontsize = 14, fontweight='bold')
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
    plt.savefig(filename, format = "svg")


