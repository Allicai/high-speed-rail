import heapq
from geopy.distance import great_circle, geodesic
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import itertools


# --------------------------------------------------- Methods/Functionality ------------------------------------------ #

class BetterPriorityQueue:
    """
      Implements a BETTER priority queue data structure designed for my UCS implementation. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, oldItem, newItem, newPriority):
        # If item not in priority queue, do the same thing as self.push
        # If item already in priority queue with equal or lower priority, do nothing
        # If item already in priority queue, replace it with the new item with the new priority
        for index, (p, c, i) in enumerate(self.heap):
            if i == oldItem:
                if p <= newPriority:
                    break
                del self.heap[index]
                self.heap.append((newPriority, c, newItem))
                heapq.heapify(self.heap)
                break
        else:
            self.push(newItem, newPriority)

    def findNodeWithState(self, state):
        for index, (p, c, i) in enumerate(self.heap):
            if i[0] == state:
                return i
        return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def great_circle_two_points(pt1: Point, pt2: Point):
    return great_circle((pt1.y, pt1.x), (pt2.y, pt2.x))


def getPassServed(city1, city2):
    pass

def getEmissionsSaved(city1, city2):
    pass

def getRailCost(city1, city2):
    pass

def getTimeSaved(city1, city2):
    pass