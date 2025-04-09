

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np

from points import gower_distance, Point, Distance
from agglomerative import complete, average, weighted_average, ward
from metrics import silhouette_index, dunn_index, ch_index
from experiment_functions import kmeans_exp, agglomerative_exp, dbscan_exp, table_plot
from optimal_n_clusters import total_dissimilarity, elbow_method, metric_optimal_n_clusters, gap_statistic



def classify_variables(df: pd.DataFrame) -> tuple[list: bool, dict[int, tuple[int, int] | tuple[float, float]]]:
    """
    Given a dataframe, returns a list of boolean indicating which variables are categorical or binary, and which not,
    along with a dictionary mapping continuous variables to their minimum and maximum values.

    Parameters
    ----------
    df: dataframe.

    Returns
    -------
    A list of boolean indicating which variables are categorical or binary (True), and which are continuos (False).
    A dictionary mapping continuous variables to theirs minimum and maximum values.
    """
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
    """
    Reads the dataset, processes it, and returns a list of points along with the Gower distance function for the dataset.

    Returns
    -------
    The list of points and the Gower distance function for the dataset.
    """
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


def elbow_exp() -> None:
    """
    Apply the Elbow method to find the optimal number of clusters for the dataset. It saves a plot as a svg file.

    """
    data, gower = reading_data_and_gower()
    max_k = 10
    elbow_method(data, max_k, 0.01, 100, 'elbow_customers.svg', total_dissimilarity, gower)


def silhouette_exp() -> None:
    """
    It saves a plot with the k (<= max_k) values and the values of Silhouette index after kmeans result.
    Done to find the optimal number of clusters for the dataset.

    """
    data, gower = reading_data_and_gower()
    max_k = 10
    metric_optimal_n_clusters(data, max_k, 0, 100, 'silhouette_customers.svg',
                              silhouette_index, gower)

def dunn_exp() -> None:
    """
    It saves a plot with the k (<= max_k) values and the values of Dunn index after kmeans result.
    Done to find the optimal number of clusters for the dataset.

    """
    data, gower = reading_data_and_gower()
    max_k = 10
    metric_optimal_n_clusters(data, max_k, 0, 100, 'dunn_customers.svg', dunn_index, gower)


def ch_exp() -> None:
    """
    It saves a plot with the k (<= max_k) values and the values of Calinski-Harabasz index after kmeans result.
    Done to find the optimal number of clusters for the dataset.

    """
    data, gower = reading_data_and_gower()
    max_k = 10
    metric_optimal_n_clusters(data, max_k, 0, 100, 'ch_customers.svg', ch_index, gower)


def gap_exp() -> int:
    """
    Gap statistic method on the dataset to find the optimal number of clusters. It saves a plot and returns the
    optimal number of clusters.

    Returns
    -------
    Optimal number of clusters.
    """
    df = pd.read_csv('customer_dataset.csv', dayfirst=True)
    df = df.dropna()
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    df['Dt_Customer'] = df['Dt_Customer'].dt.year
    df = df.drop('ID', axis=1)
    bin_or_cat, min_max = classify_variables(df)

    def gower(point1: Point, point2: Point) -> float:
        return gower_distance(point1, point2, bin_or_cat, min_max)

    return gap_statistic(df, 7, 0.01, 100, 25, 'gap_customers.svg',
                      total_dissimilarity, gower)


def subplot_dendrogram(list_linkage_matrix: list[np.ndarray], list_methods: list[str],
                       filename: str) -> None:
    """
    Given the list of linkage matrices and the list of their linkage methods,
    saves all the dendrograms in a svg file.

    Parameters
    ----------
    list_linkage_matrix: list with linkage matrices.
    list_methods: list with method names of each linkage.
    filename: name of the svg file to save the dendrograms.
    """
    fig, axes = plt.subplots(7, 1, figsize = (20, 4*7))
    for i in range(len(list_linkage_matrix)):
        dendrogram(list_linkage_matrix[i], ax = axes[i], leaf_rotation=90, leaf_font_size=3)
        axes[i].set_title(list_methods[i])
        axes[i].set_xlabel("cluster indexes")
        axes[i].set_ylabel("distance between clusters")
    plt.subplots_adjust(hspace = 0.5)
    plt.savefig(filename, format="svg")


def main():
    """
    Experiment with customers dataset. It saves a file with all the dendrograms of agglomerative clustering,
    and a file with the table of all the clustering results.

    """
    data, gower = reading_data_and_gower()
    results = []

    kmeans_result1 = kmeans_exp(data, 4, 0.01, 100, gower)
    results.append(kmeans_result1)
    kmeans_result2 = kmeans_exp(data, 3, 0.01, 100, gower)
    results.append(kmeans_result2)

    agglomerative_result1, linkage_matrix1 = agglomerative_exp(data, complete, 0.45, dist = gower)
    agglomerative_result2, linkage_matrix2 = agglomerative_exp(data, average, 0.31, dist = gower)
    agglomerative_result3, linkage_matrix3 = agglomerative_exp(data, average, 0.32, dist = gower)
    agglomerative_result4, linkage_matrix4 = agglomerative_exp(data, average, 0.33, dist = gower)
    agglomerative_result5, linkage_matrix5 = agglomerative_exp(data, weighted_average, 0.29, dist = gower)
    agglomerative_result6, linkage_matrix6 = agglomerative_exp(data, ward, 20, dist = gower)
    agglomerative_result7, linkage_matrix7 = agglomerative_exp(data, ward, 15, dist = gower)
    list_linkage_matrix = []
    for i in range(1,8):
        result = eval(f"agglomerative_result{i}")
        results.append(result)
        linkage_matrix = eval(f"linkage_matrix{i}")
        list_linkage_matrix.append(linkage_matrix)
    list_methods = ["Complete linkage", "Average linkage", "Average linkage", "Average linkage",
                    "Weighted average linkage", "Ward linkage", "Ward linkage"]

    dbscan_result1 = dbscan_exp(data, 0.1, 4, gower)
    dbscan_result2 = dbscan_exp(data, 0.11, 3, gower)
    dbscan_result3 = dbscan_exp(data, 0.1, 6, gower)
    dbscan_result4 = dbscan_exp(data, 0.1, 3, gower)
    for i in range(1,5):
        result = eval(f"dbscan_result{i}")
        results.append(result)

    #Plots
    subplot_dendrogram(list_linkage_matrix, list_methods, "dendrograms_customers.svg")
    table_plot(results, "Customers clustering", "results_customers.svg")


if __name__ == '__main__':
    main()








