
from vptree import VPTree

from points import Point, euclidean_distance, Distance, Cluster


def dbscan(data: list[Point], eps: float, min_points: int,
           dist: Distance = euclidean_distance) -> tuple[list[Cluster], Cluster]:
    """
    Given a set of data, epsilon and the minimum number of points to make a neighborhood,
    returns the list of clusters and the noise cluster. It uses a VPTree for an efficient research
    of a point neighborhood.

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
    O(N*log(N)) where N: number of points.

    """
    visited = {point: False for point in data}
    list_clusters = []
    noise = Cluster()
    vptree = VPTree(data, dist)
    for point in data:
        if not visited[point]:
            visited[point] = True
            list_neighbors = vptree.get_all_in_range(point, eps)
            if len(list_neighbors) >= min_points:
                cluster = Cluster({point})
                while len(list_neighbors) > 0:
                    (_,neighbour) = list_neighbors.pop()
                    if not(visited[neighbour]):
                        visited[neighbour] = True
                        if neighbour in noise.points:
                            noise.quit_point(neighbour)
                        cluster.add_point(neighbour)
                        new_list_neighbors = vptree.get_all_in_range(neighbour, eps)
                        if len(new_list_neighbors) >= min_points:
                            list_neighbors.extend(new_list_neighbors)
                list_clusters.append(cluster)
            else:
                noise.add_point(point)
    return list_clusters, noise




