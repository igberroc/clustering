
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from typing import Callable
import numpy as np
from time import perf_counter

from points import gower_distance, Point, Distance, euclidean_distance
from agglomerative import single, complete, average, weighted_average, median, ward, centroid, agglomerative
from metrics import silhouette_index, c_index, dunn_index
from dbscan import dbscan


def classify_variables(df: pd.DataFrame) -> tuple[list: bool, dict[int, tuple[int, int] | tuple[float, float]]]:
    bin_or_cat = [False for _ in range(len(df.columns))]
    min_max = {}
    min_values = list(df.min())
    max_values = list(df.max())
    for i in range(len(df.columns)):
        col = df.columns[i]
        column_type = df[col].dtype
        unique_values = df[col].nunique()
        if column_type == 'object' or column_type.name == 'category' or unique_values == 2 or unique_values == 1:
            bin_or_cat[i] = True
        else:
            min_max[i] = (min_values[i].item(), max_values[i].item())
    return bin_or_cat, min_max

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
    A string with the name of the algorithm and its parameters, the Silhouette index,
    C-index, Dunn index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    (linkage_matrix, list_clusters) = agglomerative(data, method, max_dist, dist)
    t1 = perf_counter()
    print(len(list_clusters))
    silhouette = silhouette_index(list_clusters, dist)
    c = c_index(data, list_clusters, dist)
    dunn = dunn_index(list_clusters, dist)
    return (f'Agglomerative({method.__name__} linkage, max_dist = {max_dist}) ', silhouette, c, dunn, t1 - t0), linkage_matrix

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
    A string with the name of the algorithm and its parameters, the Silhouette index,
    C-index, Dunn index and the execution time of the algorithm.
    """
    t0 = perf_counter()
    (list_clusters, noise) = dbscan(data, eps, min_points, dist)
    t1 = perf_counter()
    print(len(list_clusters))
    print(noise.size())
    silhouette = silhouette_index(list_clusters, dist)
    c = c_index(data, list_clusters, dist)
    dunn = dunn_index(list_clusters, dist)
    return f'DBSCAN(eps = {eps}, min_points = {min_points})', silhouette, c, dunn, t1 - t0


if __name__ == '__main__':
    df = pd.read_csv('customer_dataset.csv', dayfirst = True)
    df = df.dropna()
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst = True)
    df['Dt_Customer'] = df['Dt_Customer'].dt.year
    df = df.drop('ID', axis = 1)
    bin_or_cat, min_max = classify_variables(df)


    def gower(point1: Point, point2: Point) -> float:
        return gower_distance(point1, point2, bin_or_cat, min_max)

    df_tuples = df.itertuples(index=False, name=None)
    data = []
    for point_coordinates in df_tuples:  # Changing data into points.
        data.append(Point(*point_coordinates))
    results = []

    """
    agglomerative_result, linkage_matrix = agglomerative_exp(data, complete, 0.45, dist = gower)
    print(agglomerative_result)

    plt.figure()
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=3)
    plt.show()
    """

    dbscan_result = dbscan_exp(data, 0.12, 400, dist = gower)
    print(dbscan_result)







