import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from sklearn.preprocessing import StandardScaler

from points import gower_distance, Point, Distance
from agglomerative import complete, average, weighted_average, ward, single, agglomerative
from metrics import silhouette_index, dunn_index, ch_index
from experiment_functions import kmeans_exp, agglomerative_exp, dbscan_exp, dbscan_vptree_exp, table_plot
from optimal_n_clusters import total_dissimilarity, elbow_method, metric_optimal_n_clusters, gap_statistic
from dbscan import dbscan
from dbscan_vptree import dbscan as dbscan_vptree


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

def main1():
    df = pd.read_csv('wine_dataset.csv')
    scaler = StandardScaler()
    features = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids',
                'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']
    df[features] = scaler.fit_transform(df[features])
    df_tuples = df.itertuples(index=False, name=None)
    data = []
    for point_coordinates in df_tuples:  # Changing data into points.
        data.append(Point(*point_coordinates))

    list_clusters, noise = dbscan_vptree(data, 2.16, 3)
    for i in range(3):
        print(list_clusters[i].size())
    print(noise.size())

def main2():

    data, gower = reading_data_and_gower()


    linkage_matrix, list_clusters = agglomerative(data, complete, 0.495, gower)
    print(len(list_clusters))
    """
    result = dbscan_vptree_exp(data, 0.12, 3, gower)
    print(result)
    """

def main3():
    df = pd.read_csv('wine_dataset.csv')
    df_tuples = df.itertuples(index = False, name = None)
    data = []
    for point_coordinates in df_tuples:        #Changing data into points.
        data.append(Point(*point_coordinates))

    kmeans_results = kmeans_exp(data, 3, 0.001, 100)
    results = [kmeans_results]*17
    table_plot(results, "Prueba", "pruebas/tabla_prueba.svg")

main3()

