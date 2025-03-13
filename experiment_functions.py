
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable

from points import Cluster, euclidean_distance, Point, Distance
from kmeans import kmeans
from metrics import silhouette_index, db_index, c_index, ch_index, dunn_index
from agglomerative import agglomerative
from fuzzy import fuzzy_cmeans
from dbscan import dbscan
from EM import em


def kmeans_exp(data: list[Point], k: int, eps: float, max_iter: int
                , dist: Distance = euclidean_distance) -> tuple[str, float, float, float, float, float, float]:
    """
    Test the K-means algorithm with the given parameters.

    Parameters
    ----------
    data: list of points.
    k: number of clusters.
    eps: tolerance.
    max_iter: maximum number of iterations.
    dist: distance function.

    Returns
    -------
    A string with the name of the algorithm and its parameters, the Silhouette index, the
    Davies-Bouldin index, the C-index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    list_clusters = kmeans(data, k, eps, max_iter, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    ch = ch_index(list_clusters)
    dunn = dunn_index(list_clusters)
    return f'KMeans(eps = {eps}, max_iter = {max_iter})', silhouette, db, c, ch, dunn, t1 - t0

def agglomerative_exp(data: list[Point], method: Callable[..., float], max_dist: int = 0,
                       dist: Distance = euclidean_distance)\
        -> tuple[tuple[str, float, float, float, float, float, float], np.ndarray]:
    """
    Test the agglomerative algorithm with the given parameters.

    Parameters
    ----------
    data: list of points.
    method: linkage method.
    max_dist: maximum distance allowed to combine two clusters.
    dist: distance function.

    Returns
    -------
    A string with the name of the algorithm and its parameters, the Silhouette index, the
    Davies-Bouldin index, the C-index and the execution time of the algorithm; and the linkage matrix.
    """
    t0 = perf_counter()
    (linkage_matrix, list_clusters) = agglomerative(data, method, max_dist, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    ch = ch_index(list_clusters)
    dunn = dunn_index(list_clusters)
    return (f'Agglomerative({method.__name__} linkage, max_dist = {max_dist}) ', silhouette, db, c, ch, dunn, t1 - t0), linkage_matrix

def fuzzy_exp(data: list[Point], initial_centroids: list[Point], m: float, c: int,
               eps: float, max_iter: int, dist: Distance = euclidean_distance)\
        -> tuple[str, float, float, float, float, float, float]:
    """
    Test the fuzzy algorithm with the given parameters.

    Parameters
    ----------
    data: list of points.
    initial_centroids: list of initial centroids.
    m: fuzzification parameter.
    c: number of clusters.
    eps: tolerance.
    max_iter: maximum number of iterations.
    dist: distance function.

    Returns
    -------
    A tuple of a string with the name of the algorithm and its parameters, the Silhouette index, the
    Davies-Bouldin index, the C-index and the execution time of the algorithm.
    """
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
    ch = ch_index(list_clusters)
    dunn = dunn_index(list_clusters)
    return f'Fuzzy(m = {m}, eps = {eps}, max_iter = {max_iter})', silhouette, db, c, ch, dunn, t1 - t0

def dbscan_exp(data: list[Point], eps: float, min_points: int,
                dist: Distance = euclidean_distance) -> tuple[str, float, float, float, float, float, float]:
    """
    Test the DBSCAN algorithm with the given parameters.

    Parameters
    ----------
    data: list of points.
    eps: epsilon value for DBSCAN.
    min_points: minimum number of points to make a neighborhood.
    dist: distance function.

    Returns
    -------
    A string with the name of the algorithm and its parameters, the Silhouette index, the
    Davies-Bouldin index, the C-index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    (list_clusters, noise) = dbscan(data, eps, min_points, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    ch = ch_index(list_clusters)
    dunn = dunn_index(list_clusters)
    return f'DBSCAN(eps = {eps}, min_points = {min_points})', silhouette, db, c, ch, dunn, t1 - t0

def em_exp(data: list[Point], n_clusters: int, initial_covariances: list[np.ndarray],
            eps: float, max_iter: int) -> tuple[str, float, float, float, float, float, float]:
    """
    Test the EM algorithm with the given parameters.

    Parameters
    ----------
    data: list of points.
    n_clusters: number of clusters.
    initial_covariances: list of initial covariances matrices.
    eps: tolerance.
    max_iter: maximum number of iterations.

    Returns
    -------
    A string with the name of the algorithm and its parameters, the Silhouette index, the
    Davies-Bouldin index, the C-index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    list_clusters = em(data,n_clusters, initial_covariances, eps ,max_iter)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters)
    db = db_index(list_clusters)
    c = c_index(data, list_clusters)
    ch = ch_index(list_clusters)
    dunn = dunn_index(list_clusters)
    return f'EM(eps = {eps}, max_iter = {max_iter})', silhouette, db, c, ch, dunn, t1 - t0

def table_plot(results: list[list], plot_title: str, filename: str):
    """
    Plot a table of clustering results.

    Parameters
    ----------
    results: list of clustering results, given by experiment functions.
    plot_title: title of the plot.
    filename: filename to save the plot.

    """
    df_results = pd.DataFrame(results,
                              columns=["Algorithm", "Silhouette", "Davies-Bouldin", "C-index", "Calinski-Harabasz", "Dunn", "Time(s)"])
    plt.figure(figsize=(12, 4))
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


