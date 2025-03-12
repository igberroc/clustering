# -*- coding: utf-8 -*-


import math
from typing import Self, Callable, TypeVar


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

    def centroid(self) -> Point:
        dimension = self.points_dimension()
        s = Point.null_point(dimension)
        for point in self.points:
            s = s.sum(point)
        n = len(self.points)
        s = s.div_num(n)
        return s
    
    def clear(self) -> None:
        self.points = set()

    def copy_cluster(self) -> Self:
        cluster = Cluster()
        cluster.points = self.points.copy()
        return cluster
    
    def size(self) -> int:
        return len(self.points)
    
    def combine(self,other) -> Self:
        cluster = self.copy_cluster()
        cluster.points = cluster.points | other.points
        return cluster
        

def decompose_x_y(cluster: Cluster) -> tuple[list[float],list[float]]:
    x = []
    y = []
    for point in cluster.points:
       x.append(point.coordinates[0])
       y.append(point.coordinates[1])
    return x, y









  