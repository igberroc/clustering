
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
from dbscan_vptree import dbscan as dbscan_vptree
from EM import em


def kmeans_exp(data: list[Point], k: int, eps: float, max_iter: int,
                dist: Distance = euclidean_distance) -> tuple[str, int, str, float, float, float, float, float, float]:
    """
    Test the K-means algorithm, or K-medoids if euclidean distance is not used, with the given parameters.

    Parameters
    ----------
    data: list of points.
    k: number of clusters.
    eps: tolerance.
    max_iter: maximum number of iterations.
    dist: distance function.

    Returns
    -------
    A string with the name of the algorithm and its parameters, the Silhouette index, Davies-Bouldin index,
    C-index, Calinksi-Harabasz index, Dunn index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    list_clusters = kmeans(data, k, eps, max_iter, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters, dist)
    db = db_index(list_clusters, dist)
    c = c_index(data, list_clusters, dist)
    ch = ch_index(list_clusters, dist)
    dunn = dunn_index(list_clusters, dist)
    if dist == euclidean_distance:
        name = f'KMeans(eps = {eps}, max_iter = {max_iter})'
    else:
        name = f'KMedoids(eps = {eps}, max_iter = {max_iter})'
    return name, k, 'N/A', silhouette, db, c, ch, dunn, t1 - t0



def agglomerative_exp(data: list[Point], method: Callable[..., float], max_dist: int = 0,
                       dist: Distance = euclidean_distance)\
        -> tuple[tuple[str, int, str, float, float, float, float, float, float], np.ndarray]:
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
    A string with the name of the algorithm and its parameters, the Silhouette index, Davies-Bouldin index,
    C-index, Calinksi-Harabasz index, Dunn index and the execution time of the algorithm; and the linkage matrix.
    """
    t0 = perf_counter()
    (linkage_matrix, list_clusters) = agglomerative(data, method, max_dist, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters, dist)
    db = db_index(list_clusters, dist)
    c = c_index(data, list_clusters, dist)
    ch = ch_index(list_clusters, dist)
    dunn = dunn_index(list_clusters, dist)
    return (f'Agglomerative({method.__name__} linkage, max_dist = {max_dist}) ', len(list_clusters), 'N/A',
            silhouette, db, c, ch, dunn, t1 - t0), linkage_matrix


def fuzzy_exp(data: list[Point], initial_centroids: list[Point], m: float, c: int,
               eps: float, max_iter: int, dist: Distance = euclidean_distance)\
        -> tuple[str, int, str, float, float, float, float, float, float]:
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
    A string with the name of the algorithm and its parameters, the Silhouette index, Davies-Bouldin index,
    C-index, Calinksi-Harabasz index, Dunn index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    membership_matrix = fuzzy_cmeans(data, initial_centroids, m, c, eps, max_iter, dist)
    t1 = perf_counter()
    list_clusters = [Cluster() for _ in range(c)]
    for i in range(len(data)):
        max_index = np.argmax(membership_matrix[:,i])
        list_clusters[max_index].add_point(data[i])
    silhouette = silhouette_index(list_clusters, dist)
    db = db_index(list_clusters, dist)
    c = c_index(data, list_clusters, dist)
    ch = ch_index(list_clusters, dist)
    dunn = dunn_index(list_clusters, dist)
    return (f'Fuzzy(m = {m}, eps = {eps}, max_iter = {max_iter})', len(list_clusters), 'N/A',
            silhouette, db, c, ch, dunn, t1 - t0)


def dbscan_exp(data: list[Point], eps: float, min_points: int,
                dist: Distance = euclidean_distance) -> tuple[str, int, int, float, float, float, float, float, float]:
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
    A string with the name of the algorithm and its parameters, the Silhouette index, Davies-Bouldin index,
    C-index, Calinksi-Harabasz index, Dunn index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    (list_clusters, noise) = dbscan(data, eps, min_points, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters, dist)
    db = db_index(list_clusters, dist)
    c = c_index(data, list_clusters, dist)
    ch = ch_index(list_clusters, dist)
    dunn = dunn_index(list_clusters, dist)
    return (f'DBSCAN(eps = {eps}, min_points = {min_points})', len(list_clusters), noise.size(),
            silhouette, db, c, ch, dunn, t1 - t0)


def dbscan_vptree_exp(data: list[Point], eps: float, min_points: int,
                dist: Distance = euclidean_distance) -> tuple[str, int, int, float, float, float, float, float, float]:
    """
    Test the DBSCAN VP-tree version algorithm with the given parameters.

    Parameters
    ----------
    data: list of points.
    eps: epsilon value for DBSCAN.
    min_points: minimum number of points to make a neighborhood.
    dist: distance function.

    Returns
    -------
    A string with the name of the algorithm and its parameters, the Silhouette index, Davies-Bouldin index,
    C-index, Calinksi-Harabasz index, Dunn index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    (list_clusters, noise) = dbscan_vptree(data, eps, min_points, dist)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters, dist)
    db = db_index(list_clusters, dist)
    c = c_index(data, list_clusters, dist)
    ch = ch_index(list_clusters, dist)
    dunn = dunn_index(list_clusters, dist)
    return (f'DBSCAN_vptree(eps = {eps}, min_points = {min_points})', len(list_clusters), noise.size(),
            silhouette, db, c, ch, dunn, t1 - t0)


def em_exp(data: list[Point], n_clusters: int, initial_covariances: list[np.ndarray],
            eps: float, max_iter: int, dist: Distance = euclidean_distance)\
        -> tuple[str, int, str, float, float, float, float, float, float]:
    """
    Test the EM algorithm with the given parameters.

    Parameters
    ----------
    data: list of points.
    n_clusters: number of clusters.
    initial_covariances: list of initial covariances matrices.
    eps: tolerance.
    max_iter: maximum number of iterations.
    dist: distance function (for metrics).

    Returns
    -------
    A string with the name of the algorithm and its parameters, the Silhouette index, Davies-Bouldin index,
    C-index, Calinksi-Harabasz index, Dunn index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    list_clusters = em(data,n_clusters, initial_covariances, eps ,max_iter)
    t1 = perf_counter()
    silhouette = silhouette_index(list_clusters, dist)
    db = db_index(list_clusters, dist)
    c = c_index(data, list_clusters, dist)
    ch = ch_index(list_clusters, dist)
    dunn = dunn_index(list_clusters, dist)
    return f'EM(eps = {eps}, max_iter = {max_iter})', n_clusters, 'N/A', silhouette, db, c, ch, dunn, t1 - t0


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
                         columns=["Algorithm", "Clusters", "Noise points", "Silhouette", "Davies-Bouldin", "C-index",
                                  "Calinski-Harabasz", "Dunn", "Time(s)"])
    row_count = len(df_results)
    col_count = 9
    row_height = 1
    col_width = 3.5
    fig_width = col_count * col_width
    fig_height = (row_count + 1) * row_height
    plt.figure(figsize=(fig_width, fig_height))
    plt.title(plot_title, fontsize=28, fontweight='bold')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    table = ax.table(
        cellText=df_results.round(3).values,
        colLabels=df_results.columns,
        cellLoc='center',
        loc='upper center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(26)
    table.auto_set_column_width([0, 1, 2, 3, 4, 5, 6, 7, 8])
    cell_height = 0.08
    for key, cell in table.get_celld().items():
        cell.set_height(cell_height)
    plt.savefig(filename, format = "svg", bbox_inches='tight')



