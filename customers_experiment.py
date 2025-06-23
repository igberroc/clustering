

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np

from points import gower_distance, Point, Distance
from agglomerative import complete, average
from metrics import silhouette_index, dunn_index, ch_index
from experiment_functions import kmeans_exp, agglomerative_exp, dbscan_exp, dbscan_vptree_exp, table_plot
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
    A list of boolean indicating which variables are categorical or binary (True), and which are continuos or
     ordinal (False).
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
    df = df.drop(['ID','Z_CostContact', 'Z_Revenue'], axis = 1)
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


def plot_dendrogram(linkage_matrix: np.ndarray, method: str, filename: str) -> None:
    """
    Given a linkage matrix and the linkage method, saves the dendrogram in a svg file.
    """
    plt.figure(figsize=(20, 7))
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=5)
    plt.title(f"Dendrogram ({method})")
    plt.xlabel("Cluster indexes")
    plt.ylabel("Distance between clusters")
    plt.tight_layout()
    plt.savefig(filename, format="svg")
    plt.close()

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
    agglomerative_result2, linkage_matrix2 = agglomerative_exp(data, complete, 0.495, dist=gower)
    agglomerative_result3, linkage_matrix3 = agglomerative_exp(data, complete, 0.51, dist = gower)
    agglomerative_result4, linkage_matrix4 = agglomerative_exp(data, average, 0.32, dist=gower)
    agglomerative_result5, linkage_matrix5 = agglomerative_exp(data, average, 0.34, dist=gower)
    agglomerative_result6, linkage_matrix6 = agglomerative_exp(data, average, 0.37, dist = gower)


    for i in range(1,7):
        result = eval(f"agglomerative_result{i}")
        results.append(result)

    dbscan_result1 = dbscan_exp(data, 0.1, 6, gower)
    dbscan_result2 = dbscan_exp(data, 0.12, 3, gower)
    dbscan_result3 = dbscan_exp(data, 0.1, 4, gower)
    dbscan_result4 = dbscan_exp(data, 0.11, 3, gower)


    for i in range(1,5):
        result = eval(f"dbscan_result{i}")
        results.append(result)
    dbscan_result1 = dbscan_vptree_exp(data, 0.1, 6, gower)
    dbscan_result2 = dbscan_vptree_exp(data, 0.12, 3, gower)
    dbscan_result3 = dbscan_vptree_exp(data, 0.1, 4, gower)
    dbscan_result4 = dbscan_vptree_exp(data, 0.11, 3, gower)


    for i in range(1,5):
        result = eval(f"dbscan_result{i}")
        results.append(result)

    #Plots
    plot_dendrogram(linkage_matrix1, "Complete linkage", "complete_aglom_customers.svg")
    plot_dendrogram(linkage_matrix4, "Average linkage", "average_aglom_customers.svg")
    table_plot(results, "Customers clustering", "results_customers.svg")


if __name__ == '__main__':
    main()








