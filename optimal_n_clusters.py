

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable
import random
import math

from points import Cluster, euclidean_distance, Point, Distance
from kmeans import kmeans


def inertia(list_clusters: list[Cluster], dist: Distance = euclidean_distance) -> float:
    """
    Given a list of clusters and a distance function, returns the inertia metric.

    Parameters
    ----------
    list_clusters: list of clusters.
    dist: distance function (for inertia is usually euclidean distance).

    Returns
    -------
    Inertia metric.
    """
    inertia = 0
    for cluster in list_clusters:
        centroid = cluster.centroid(dist)
        for point in cluster.set_points():
            inertia += dist(centroid, point)**2
    return inertia

def total_dissimilarity(list_clusters: list[Cluster], dist: Distance) -> float:
    """
    Given a list of clusters and the distance function, computes the total dissimilarity.

    Parameters
    ----------
    list_clusters: list of clusters.
    dist: distance function.

    Returns
    -------
    Total dissimilarity (sum of distances between each point and their medoid).
    """
    total_dissimilarity = 0
    for cluster in list_clusters:
        medoid = cluster.centroid(dist)
        for point in cluster.set_points():
            total_dissimilarity += dist(medoid, point)
    return total_dissimilarity

def elbow_method(data: list[Point], max_k: int, eps: float, max_iter: int,
                 filename: str, method: Callable[...,float] = inertia,
                 dist: Distance = euclidean_distance) -> None:
    """
    Given a list of points, the maximum k value for finding the optimal number of clusters,
    the parameters for k-means, the filename to save the plot, and the criterion for evaluating the elbow method,
    it saves the plot.

    Parameters
    ----------
    data: list of points.
    max_k: maximum k value for finding the optimal number of clusters.
    eps: tolerance for k-means.
    max_iter: maximum number of iterations for k-means.
    filename: filename to save the plot.
    method: metric function for evaluating the elbow method.
    dist: distance function.

    """
    k_values = range(1, max_k + 1)
    method_values = []
    for k in k_values:
        list_clusters = kmeans(data, k, eps, max_iter, dist)
        method_values.append(method(list_clusters, dist))
    plt.figure()
    plt.plot(k_values, method_values, marker = 'o', linestyle = '-', color = 'b')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(f'{method.__name__}')
    plt.savefig(filename, format = 'svg')

def metric_optimal_n_clusters(data: list[Point], max_k: int, eps: float, max_iter: int,
                      filename: str, metric: Callable[..., float], dist: Distance = euclidean_distance) -> None:
    """
    Given a list of points, the maximum k value for finding the optimal number of clusters,
    the parameters for k-means and the filename to save the plot, it saves the plot of the k values and
    the values of the metric after kmeans result.

    Parameters
    ----------
    data: list of points.
    max_k: maximum k value for finding the optimal number of clusters.
    eps: tolerance for k-means.
    max_iter: maximum number of iterations for k-means.
    filename: filename to save the plot.
    dist: distance function.

    """
    k_values = range(2, max_k + 1)
    metric_values = []
    for k in k_values:
        list_clusters = kmeans(data, k, eps, max_iter, dist)
        if metric.__name__ == 'c_index':
            value = metric(data, list_clusters, dist)
        else:
            value = metric(list_clusters, dist)
        metric_values.append(value)
    plt.figure()
    plt.plot(k_values, metric_values, marker = 'o', linestyle = '-', color = 'b')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(metric.__name__)
    plt.savefig(filename, format='svg')

def random_data_sample(df: pd.DataFrame) -> list[Point]:
    """
    Given a dataframe, returns a random sample of points for the GAP statistic. For continuous variables,
    values are randomly sampled from a uniform interval defined by the minimum and maximum values of
    each variable. For categorical and binary variables, the sampled data preserves the same proportion
    of categories as in the original data.

    Parameters
    ----------
    df: dataframe.

    Returns
    -------
    list of random points.
    """
    variables_ranges = []
    for i in range(len(df.columns)):
        col = df.columns[i]
        column_type = df[col].dtype
        unique_values = df[col].nunique()
        if column_type == 'object' or column_type.name == 'category' or unique_values == 2 or unique_values == 1:
            category_counts = df[col].value_counts(normalize = True)
            categories = category_counts.index.tolist()
            probabilities = category_counts.values.tolist()
            variables_ranges.append((categories, probabilities))
        else:
            min_val = df[col].min()
            max_val = df[col].max()
            min_val = min_val.item() if isinstance(min_val, (np.integer, np.floating)) else min_val
            max_val = max_val.item() if isinstance(max_val, (np.integer, np.floating)) else max_val
            variables_ranges.append([min_val, max_val])
    n = len(df)
    sample = [0 for _ in range(n)]
    for i in range(n):
        point_coordinates = []
        for ranges in variables_ranges:
            if type(ranges) == list:
                if type(ranges[0]) == int:
                    point_coordinates.append(random.randint(ranges[0], ranges[1]))
                else:
                    point_coordinates.append(random.uniform(ranges[0], ranges[1]))
            else:
                categories, probabilities = ranges
                point_coordinates.append(np.random.choice(categories, p = probabilities).item())
        sample[i] = Point(*tuple(point_coordinates))
    return sample

def gap_array(df: pd.DataFrame, max_k: int, eps: float, max_iter: int, n_samples: int,
       method: Callable[...,float] = inertia, dist: Distance = euclidean_distance) -> tuple[list[float], list[float]]:
    """
    Given a dataframe, max k for finding the optimal number of clusters, eps and max_iter parameters for kmeans,
    the number of samples for gap statistic, and the metric for evaluating the gap statistic, returns one list
    with the gap values and other list with the standard errors of the gap for each k.

    Parameters
    ----------
    df: dataframe.
    max_k: maximum k value for finding the optimal number of clusters.
    eps: tolerance for k-means.
    max_iter: maximum number of iterations for k-means.
    n_samples: number of samples for gap statistic.
    method: metric function for evaluating the gap statistic.
    dist: distance function.

    Returns
    -------
    The list of gap values and the list of standard errors of the gap for each k.
    """
    df_tuples = df.itertuples(index = False, name = None)
    data = [Point(*point_coordinates) for point_coordinates in df_tuples]
    list_samples = [random_data_sample(df) for _ in range(n_samples)]
    list_gap = [0 for _ in range(max_k)]
    list_std_errors = [0 for _ in range(max_k)]
    print("Starting GAP statistic")
    for k in range(1, max_k + 1):
        print("k:", k)
        list_clusters = kmeans(data, k, eps, max_iter, dist)
        w_k = method(list_clusters, dist)
        log_wk = np.log(w_k)
        log_wk_samples = np.array([])
        for sample in list_samples:
            list_clusters = kmeans(sample, k, eps, max_iter, dist)
            w_kb = method(list_clusters, dist)
            log_wk_samples = np.append(log_wk_samples, np.log(w_kb))
        mean_log_wkb = np.mean(log_wk_samples)
        list_gap[k - 1] = mean_log_wkb - log_wk
        if k != 1:
            std_error = np.std(log_wk_samples)
            list_std_errors[k - 1] = std_error * math.sqrt(1 + 1/n_samples)
    return list_gap, list_std_errors

def gap_statistic(df: pd.DataFrame, max_k: int, eps: float, max_iter: int, n_samples: int, filename: str,
                  method: Callable[...,float] = inertia, dist: Distance = euclidean_distance) -> int:
    """
    Given a dataframe, max k for finding the optimal number of clusters, eps and max_iter parameters for kmeans,
    the number of samples for gap statistic, and the metric for evaluating the gap statistic, it saves a plot
    with the gap values and returns the optimal number of clusters according to the gap statistic.

    Parameters
    ----------
    df: dataframe.
    max_k: maximum k value for finding the optimal number of clusters.
    eps: tolerance for k-means.
    max_iter: maximum number of iterations for k-means.
    n_samples: number of samples for gap statistic.
    filename: filename to save the plot of gap values.
    method: metric function for evaluating the gap statistic.
    dist: distance function.

    Returns
    -------
    The optimal number of clusters according to the gap statistic.
    """
    k_values = range(1, max_k + 1)
    list_gap, list_std_error = gap_array(df, max_k, eps, max_iter, n_samples, method, dist)
    plt.figure()
    plt.plot(k_values, list_gap, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('GAP')
    plt.savefig(filename, format = 'svg')
    k = 1
    while k < max_k and list_gap[k - 1] < list_gap[k] - list_std_error[k]:
        k += 1
    return k


