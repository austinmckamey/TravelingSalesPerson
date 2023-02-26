#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self.total = 0
        self.pruned = 0
        self.ncities = 0
        self.states = []
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def greedy(self, time_allowance=60.0):
        pass

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

    def branchAndBound(self, time_allowance=60.0):  # O(n^2*logn*2^n) (time)
        start_time = time.time()
        results = self.defaultRandomTour(time_allowance)  # O(nb^n) (time)
        bssf = results['soln']
        bssf.cost = np.inf

        self.total = 1
        self.pruned = 0
        max_size = 1
        count = 0
        cities = self._scenario.getCities()
        self.ncities = len(cities)
        curr_matrix = np.zeros((self.ncities, self.ncities))

        # populate matrix with initial cost to all cities
        for i in range(self.ncities):  # O(n^2) where n is the # of cities (time)
            for j in range(self.ncities):
                curr_matrix.itemset((i, j), cities[i].costTo(cities[j]))  # O(n^2) size of every matrix (space)
        initial_matrix, lower_bound = self.lowerBound(curr_matrix)  # O(n) (time)
        initial = Node(initial_matrix, [0])

        # priority queue is sorted by lowest depth in tree, then lower bound, and finally node id
        heapq.heappush(self.states, (self.ncities, lower_bound, initial))  # O(logn) (time)
                                                                           # O(b^n) avg queue size (space)

        while len(self.states) > 0 and time.time() - start_time < time_allowance:  # O(b^n) where b is the avg # of
                                                                                   # nodes put on the queue expanding (time)
            tuple_p = heapq.heappop(self.states)  # O(logn) (time)
            if tuple_p[1] < bssf.cost:
                list_t = self.expand(tuple_p)  # O(n^2) (time) O(tn^2) where t is the number of branches plus the matrix (space)
                for i in range(len(list_t)):  # O(n) where n is the # of new states (time)
                    temp = self.test(list_t[i])
                    if temp < bssf.cost:
                        route = []
                        for j in range(len(list_t[i][2].id)):  # O(n) where n is the # of cities (time)
                            route.append(cities[list_t[i][2].id[j]])
                        bssf = TSPSolution(route)
                        bssf.cost = temp
                        count += 1
                    elif list_t[i][1] < bssf.cost:
                        heapq.heappush(self.states, list_t[i])  # O(logn) (time)
                        if len(self.states) > max_size:
                            max_size = len(self.states)
                    else:
                        self.pruned += 1
            else:
                self.pruned += 1
        self.pruned += len(self.states)
        end_time = time.time()

        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = max_size
        results['total'] = self.total
        results['pruned'] = self.pruned
        return results

    # computes the lower bound and reduced cost matrix of the state
    def lowerBound(self, cost_matrix, prev_bound=0):  # O(n) where n is the # of cities (time & space)
        lower_bound = prev_bound
        row_min_values = np.min(cost_matrix, 1)[:, None]  # O(n) (time & space)
        for i in range(self.ncities):  # O(n) (time)
            if row_min_values[i][0] == np.inf:
                row_min_values[i][0] = 0
            lower_bound += row_min_values[i][0]  # O(1) (time & space)
        row_min_matrix = cost_matrix - row_min_values
        col_min_values = np.min(row_min_matrix, 0)[None, :]  # O(n) (time & space)
        for i in range(self.ncities):  # O(n) (time)
            if col_min_values[0][i] == np.inf:
                col_min_values[0][i] = 0
            lower_bound += col_min_values[0][i]  # O(1) (time & space)
        min_cost_matrix = row_min_matrix - col_min_values
        return min_cost_matrix, lower_bound

    # computes the valid branches coming off of the current state
    def expand(self, p):  # O(n^2) (time)
        list_t = []  # O(b) where b is the avg # of new states created (space)
        for i in range(self.ncities):  # O(n) where n is # of cities (time)
            arr = p[2].id.copy()
            first = arr[len(arr) - 1]
            # creates copy of matrix to reduce
            matrix = np.copy(p[2].cost_matrix)  # O(n^2) (space)
            prev = matrix[first, i]
            matrix[first] = np.inf
            matrix[:, i] = np.inf
            reduced_matrix, l_bound = self.lowerBound(matrix, p[1] + prev)  # O(n) (time)
            # removing possibilities of doubling back to a previous city
            if not arr.__contains__(i):
                self.total += 1
                # pruning infinite bounds immediately
                if l_bound != np.inf:
                    arr.append(i)
                    node = Node(reduced_matrix, arr)
                    list_t.append((p[0] - 1, l_bound, node))  # O(1) (time)
                else:
                    self.pruned += 1
        return list_t

    # checks whether the tour has reached the bottom of the tree
    def test(self, t):
        if t[0] == 1:  # O(1) (time)
            return t[1]
        else:
            return np.inf

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        pass


# stores the cost matrix of state and id, which is an array cities in tour
class Node:
    def __init__(self, c_m, c):
        self.cost_matrix = c_m
        self.id = c

    def __lt__(self, other):
        return self.id[0] < other.id[0]
