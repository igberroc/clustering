

import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from scipy.cluster.hierarchy import dendrogram
from typing import Callable
import numpy as np
from time import perf_counter

from points import gower_distance, Point, Distance, euclidean_distance
from agglomerative import single, complete, average, weighted_average, median, ward, centroid, agglomerative
from metrics import silhouette_index, c_index, dunn_index, ch_index
from kmeans import kmeans
from dbscan import dbscan
from experiment_functions import kmeans_exp, agglomerative_exp, dbscan_exp, em_exp, table_plot
from optimal_n_clusters import (total_dissimilarity, elbow_method, metric_optimal_n_clusters, random_data_sample,
                                gap_statistic)


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


def reading_data_and_gower() -> tuple[list[Point], Distance]:
    df = pd.read_csv('customer_dataset.csv', dayfirst = True)
    df = df.dropna()
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst = True)
    df['Dt_Customer'] = df['Dt_Customer'].dt.year
    df = df.drop('ID', axis = 1)
    bin_or_cat, min_max = classify_variables(df)

    def gower(point1: Point, point2: Point) -> float:
        return gower_distance(point1, point2, bin_or_cat, min_max)

    df_tuples = df.itertuples(index = False, name = None)
    data = []
    for point_coordinates in df_tuples:  # Changing data into points.
        data.append(Point(*point_coordinates))
    return data, gower


def elbow_exp():
    data, gower = reading_data_and_gower()
    max_k = 10
    elbow_method(data, max_k, 0.01, 100, 'elbow_customers.svg', total_dissimilarity, gower)


def silhouette_exp():
    data, gower = reading_data_and_gower()
    max_k = 10
    metric_optimal_n_clusters(data, max_k, 0, 100, 'silhouette_customers.svg',
                              silhouette_index, gower)

def dunn_exp():
    data, gower = reading_data_and_gower()
    max_k = 10
    metric_optimal_n_clusters(data, max_k, 0, 100, 'dunn_customers.svg', dunn_index, gower)

def ch_exp():
    data, gower = reading_data_and_gower()
    max_k = 10
    metric_optimal_n_clusters(data, max_k, 0, 100, 'ch_customers.svg', ch_index, gower)

def gap_exp():
    df = pd.read_csv('customer_dataset.csv', dayfirst=True)
    df = df.dropna()
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    df['Dt_Customer'] = df['Dt_Customer'].dt.year
    df = df.drop('ID', axis=1)



if __name__ == '__main__':
    """
    data, gower = reading_data_and_gower()
    
    
    agglomerative_result, linkage_matrix = agglomerative_exp(data, complete, 0.46, dist = gower)
    print(agglomerative_result)

    plt.figure(figsize = (20,4))
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=3)
    plt.savefig('complete_aglom_customers.svg', format='svg')
    """

    df = pd.read_csv('customer_dataset.csv', dayfirst=True)
    df = df.dropna()
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    df['Dt_Customer'] = df['Dt_Customer'].dt.year
    df = df.drop('ID', axis=1)
    bin_or_cat, min_max = classify_variables(df)

    def gower(point1: Point, point2: Point) -> float:
        return gower_distance(point1, point2, bin_or_cat, min_max)

    print(gap_statistic(df, 7, 0.01, 100, 10, 'gap_customers.svg',
                        total_dissimilarity, gower))










