from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import random
import numpy as np

from points import Point
from agglomerative import median, complete, average, ward, weighted_average, centroid, single
from experiment_functions import kmeans_exp, agglomerative_exp, fuzzy_exp, dbscan_exp, em_exp, table_plot

df = pd.read_csv('wine_dataset.csv')
scaler = StandardScaler()
features = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids',
            'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']
df[features] = scaler.fit_transform(df[features])
df_tuples = df.itertuples(index=False, name=None)
data = []
for point_coordinates in df_tuples:
    data.append(Point(*point_coordinates))

"""
kmeans_results = kmeans_exp(data, 3, 0.001, 100)
print(kmeans_results)
"""

"""
agglomerative_results1, linkage_matrix1 = agglomerative_exp(data, ward, 20)
print(agglomerative_results1)

plt.figure()
dendrogram(linkage_matrix1, leaf_rotation = 90, leaf_font_size = 3)
plt.show()

"""

"""
initial_centroids = [0 for _ in range(3)]
for i in range(3):
    point_coordinates = tuple([random.uniform(-2, 2) for _ in range(13)])
    initial_centroids[i] = Point(*point_coordinates)
fuzzy_results = fuzzy_exp(data, initial_centroids, 1.1, 3, 0.001, 100)
print(fuzzy_results)
"""

"""
dbscan_results = dbscan_exp(data, 2.15, 10)
print(dbscan_results)
"""

"""
mean = df.mean()
variance = df.var()
print(mean)
print(variance)
"""

"""
initial_covariances = [1000*np.eye(13) for _ in range(3)]
em_results = em_exp(data, 3, initial_covariances, 1e-20, 100)
print(em_results)
"""