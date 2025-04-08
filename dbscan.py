

from points import Point, euclidean_distance,Distance,Cluster

def neighbors(data: list[Point], point: Point, eps: float,
            dist: Distance = euclidean_distance) -> list[Point]:
    """
    Given a set of data, a point and epsilon, returns the list of neighbors.

    Parameters
    ----------
    data : list of points.
    point : point to find neighbors.
    eps : epsilon value for dbscan.
    dist : distance to use.

    Returns
    -------
    The list of neighbors.

    """
    list_neighbors = []
    for p in data:
        if dist(p, point) <= eps and p != point:
            list_neighbors.append(p)
    return list_neighbors


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
    visited = {point: False for point in data}
    list_clusters = []
    noise = Cluster()
    for point in data:
        if not(visited[point]):
            visited[point] = True
            list_neighbors = neighbors(data, point, eps, dist)
            if len(list_neighbors) + 1 >= min_points:
                cluster = Cluster({point})
                while len(list_neighbors) > 0:
                    neighbour = list_neighbors.pop()
                    if not(visited[neighbour]):
                        visited[neighbour] = True
                        if neighbour in noise.points:
                            noise.quit_point(neighbour)
                        cluster.add_point(neighbour)
                        new_list_neighbors = neighbors(data, neighbour, eps, dist)
                        if len(new_list_neighbors) + 1 >= min_points:
                            list_neighbors = list_neighbors + new_list_neighbors
                list_clusters.append(cluster)
            else:
                noise.add_point(point)
    return list_clusters, noise





                    
                