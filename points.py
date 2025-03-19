# -*- coding: utf-8 -*-

import numpy as np
import math
from typing import Self, Callable, TypeVar

from scipy.spatial import distance_matrix


class Point:
    def __init__(self, *coord: float):
        self.coordinates = coord
    
    def dimension(self) -> int:
        return len(self.coordinates)
    
    def get_coordinates(self) -> tuple():
        return self.coordinates
    
    def sum(self, other: Self) -> Self:
        s = [0 for _ in range(self.dimension())]
        for i in range(self.dimension()):
            s[i] = self.coordinates[i] + other.coordinates[i]
        return Point(*tuple(s))
    
    def mul_num(self, n:int) -> Self:
        coord = list(self.coordinates)
        for i in range(self.dimension()):
            coord[i] = coord[i] * n
        return Point(*tuple(coord))
            
    def div_num(self, n:int) -> Self:
        coord = list(self.coordinates)
        for i in range(self.dimension()):
            coord[i] = coord[i]/n
        return Point(*tuple(coord))
    
    @staticmethod
    def null_point(dimension) -> Self:
        p = Point()
        p.coordinates = (0,)*dimension
        return p
    

T = TypeVar('T')
Distance = Callable[[T,T], float]

     
def euclidean_distance(Point1: Point, Point2: Point) -> float:
    cord1 = Point1.get_coordinates()
    cord2 = Point2.get_coordinates()
    d = 0
    for x1,y1 in zip(cord1,cord2):
        d += (x1 - y1)**2
    return math.sqrt(d)
        
        
def manhattan_distance(Point1: Point, Point2: Point) -> float:
    cord1 = Point1.get_coordinates()
    cord2 = Point2.get_coordinates()
    d = 0
    for x1,y1 in zip(cord1,cord2):
        d += abs(x1 - y1)
    return d


def str_distance(s1: str, s2: str) -> float:
    return sum(1 for x,y in zip(s1,s2) if x != y)


def gower_distance(Point1: Point, Point2: Point, bin_or_cat: list[bool],
                   min_max: dict[int, tuple[int, int] | tuple[float, float]] ) -> float:
    """
    Given two points, a list which indicates if each variable is binary or categorical, or not, and
    a dictionary with minimum and maximum values for continuos variables.

    Parameters
    ----------
    Point1: first point.
    Point2: second point.
    bin_or_cat: list of bool which indicates if each variable is binary or categorical (True), or not (False).
    min_max: dictionary with minimum and maximum values for continuous variables.

    Returns
    -------
    Gower distance.
    """
    cord1 = Point1.get_coordinates()
    cord2 = Point2.get_coordinates()
    dim = Point1.dimension()
    d = 0
    for i in range(dim):
        if bin_or_cat[i]:
            if cord1[i] != cord2[i]:
                d += 1
        else:
            d += abs(cord1[i] - cord2[i])/(min_max[i][1] - min_max[i][0])
    return d / dim

class Cluster:
    def __init__(self, points: set[Point] = None):
        if points == None:
            self.points = set()
        else:
            self.points = points

    def set_points(self) -> set[Point]:
        return self.points
        
    def add_point(self,point: Point):
        self.points.add(point)
        
    def quit_point(self,point: Point):
        self.points.remove(point)

    def points_dimension(self) -> int:
        point = next(iter(self.points))
        return point.dimension()

    def points_sum(self) -> Point:
        dimension = self.points_dimension()
        s = Point.null_point(dimension)
        for point in self.points:
            s = s.sum(point)
        return s

    def centroid(self, dist: Distance = euclidean_distance) -> Point:
        if dist == euclidean_distance:
            dimension = self.points_dimension()
            mean = Point.null_point(dimension)
            for point in self.points:
                mean = mean.sum(point)
            n = len(self.points)
            mean = mean.div_num(n)
            return mean
        else:
            cluster_list = list(self.points)
            n = len(cluster_list)
            distance_matrix = np.zeros((n,n))
            for i in range(n):
                point1 = cluster_list[i]
                for j in range(i+1,n):
                    point2 = cluster_list[j]
                    distance_matrix[i,j] = dist(point1, point2)
            minimum = math.inf
            for i in range(n):
                intra_cluster_sum = 0
                for j in range(n):
                    if j < i:
                        intra_cluster_sum += distance_matrix[j,i]
                    if j > i:
                        intra_cluster_sum += distance_matrix[i,j]
                if intra_cluster_sum < minimum:
                    minimum = intra_cluster_sum
                    medoid = cluster_list[i]
            return medoid

    def clear(self) -> None:
        self.points = set()

    def size(self) -> int:
        return len(self.points)
    
    def combine(self,other) -> Self:
        cluster = Cluster()
        cluster.points = self.points | other.points
        return cluster
        

def decompose_x_y(cluster: Cluster) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for point in cluster.points:
       x.append(point.coordinates[0])
       y.append(point.coordinates[1])
    return x, y









  