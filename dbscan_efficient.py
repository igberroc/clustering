

from sklearn.neighbors import BallTree
import numpy as np
import matplotlib.pyplot as plt
import random

from points import Point, euclidean_distance,Distance,Cluster, decompose_x_y





def dbscan(data: list[Point], eps: float, min_points: int,
           dist: Distance = euclidean_distance) -> tuple[list[Cluster], Cluster]:
    """
    Given a set of data, epsilon and the minimum number of points to make a neighborhood,
    returns the list of clusters and the noise cluster.

    Parameters
    ----------
    data : list of points.
    eps: epsilon value for dbscan.
    min_points: minimum number of points to make a neighborhood.
    dist : distance to use.

    Returns
    -------
    The list of clusters and a cluster with noise points.

    Complexity
    -------
    O(N^2) where N: number of points.

    """
    n = len(data)
    visited = [False for _ in range(n)]
    list_clusters = []
    noise = Cluster()
    indexes = np.arange(len(data)).reshape(-1, 1)
    ball_tree = BallTree(indexes, metric = lambda index1, index2: dist(data[int(index1[0])], data[int(index2[0])]))
    for i in range(n):
        if not visited[i]:
            visited[i] = True
            point = data[i]
            neighbors_indexes = ball_tree.query_radius([[i]], r = eps)[0].tolist()
            if len(neighbors_indexes) >= min_points:
                cluster = Cluster({point})
                while len(neighbors_indexes) > 0:
                    neighbour_index = neighbors_indexes.pop()
                    if not(visited[neighbour_index]):
                        visited[neighbour_index] = True
                        neighbour = data[neighbour_index]
                        if neighbour in noise.points:
                            noise.quit_point(neighbour)
                        cluster.add_point(neighbour)
                        new_neighbors_indexes = ball_tree.query_radius([[neighbour_index]], r = eps)[0].tolist()
                        if len(new_neighbors_indexes) >= min_points:
                            neighbors_indexes = neighbors_indexes + new_neighbors_indexes
                list_clusters.append(cluster)
            else:
                noise.add_point(point)
    return list_clusters, noise


def main1():
    delta = 0.25
    n_points = 20
    data1 = [Point(1 + random.uniform(-delta, delta), 1 + random.uniform(-delta, delta)) for _ in range(n_points)]
    data2 = [Point(1 + random.uniform(-delta, delta), 0 + random.uniform(-delta, delta)) for _ in range(n_points)]
    data3 = [Point(0 + random.uniform(-delta, delta), 1 + random.uniform(-delta, delta)) for _ in range(n_points)]
    data4 = [Point(0 + random.uniform(-delta, delta), 0 + random.uniform(-delta, delta)) for _ in range(n_points)]
    data = data1 + data2 + data3 + data4

    list_clusters, noise = dbscan(data, 0.5,4)

    (x1, y1) = decompose_x_y(list_clusters[0])
    (x2, y2) = decompose_x_y(list_clusters[1])
    (x3, y3) = decompose_x_y(list_clusters[2])
    (x4, y4) = decompose_x_y(list_clusters[3])

    plt.plot(x1, y1, 'o', markerfacecolor='red')
    plt.plot(x2, y2, 'o', markerfacecolor='blue')
    plt.plot(x3, y3, 'o', markerfacecolor='green')
    plt.plot(x4, y4, 'o', markerfacecolor='yellow')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Clusters k-means")
    plt.legend()
    plt.show()

main1()