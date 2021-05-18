#!/usr/bin/python3

from numpy import inf
import copy

class Node:
    def __init__(self, label: int, distance: int, predecessor: int, importance = -1):
        self.label = label
        self.distance = distance
        self.predecessor = predecessor
        # In all cases below, importance values are equal to node labels,
        # but we can support scenarios with different values as well
        self.importance = importance

    def __repr__(self):
        string = "Node label:\t{}".format(self.label)
        if self.importance != -1:
            string += "\tImportance:\t{}".format(self.importance)
        string += "\tDistance:\t{}".format(self.distance)
        if self.predecessor is None:
            string += "\tPredecessor:\tNone"
        else:
            string += "\tPredecessor:\t{}".format(self.predecessor)
        return string

# Excercise 1
def relax(queue, u: Node, v: Node, w: int):
    if u.distance + w < v.distance:
        v.distance = u.distance + w
        build_heap(queue)
        v.predecessor = u.label

def extract_min(queue) -> Node:
    return queue.pop(0)

def is_leaf(heap, i):
    if i >= (len(heap) // 2) and i <= len(heap):
        return True
    return False

def heapify(heap, i):
    if not is_leaf(heap, i):
        if (heap[i].distance > heap[2*i].distance or
            heap[i].distance > heap[2*i+1].distance):
            if heap[2*i].distance < heap[2*i+1].distance:
                heap[i], heap[2*i] = heap[2*i], heap[i]
                heapify(heap, 2*i)
            else:
                heap[i], heap[2*i+1] = heap[2*i+1], heap[i]
                heapify(heap, 2*i+1)

def build_heap(array):
    for i in range(len(array) // 2, 0, -1):
        heapify(array, i)

def dijkstra(graph, source: int):
    nodes = [Node(i, inf, None) for i in range(0, len(graph))]
    nodes[source].distance = 0

    queue = nodes.copy()
    queue[0], queue[source] = queue[source], queue[0]

    while len(queue) > 0:
        u = extract_min(queue)
        i = u.label
        neighbors = []
        for node in nodes:
            if graph[i][node.label] > 0:
                neighbors.append(node)

        for node in neighbors:
            relax(queue, u, node, graph[i][node.label])

    return nodes

def backwards_dijkstra(graph, source: int):
    nodes = [Node(i, inf, None) for i in range(0, len(graph))]
    nodes[source].distance = 0

    queue = nodes.copy()
    queue[0], queue[source] = queue[source], queue[0]

    while len(queue) > 0:
        u = extract_min(queue)
        i = u.label
        neighbors = []
        for node in nodes:
            if graph[node.label][i] > 0:
                neighbors.append(node)

        for node in neighbors:
            relax(queue, u, node, graph[node.label][i])

    return nodes

# Exercise 2a

# matrix is the adjacency matrix, index is the 0-based index of the node
# we want to remove.
def remove_node_add_shortcut(matrix, index):
    for i in range(0, len(matrix)):
        # find all edges that connect any node
        # to the node we want to delete
        if matrix[i][index] > 0:
            for j in range(0, len(matrix[0])):
                if matrix[index][j] > 0:
                    if i != j and (matrix[i][j] == 0 or matrix[i][j] > matrix[i][index] + matrix[index][j]):
                        matrix[i][j] = matrix[i][index] + matrix[index][j]
    del matrix[index]
    for i in range(0, len(matrix)):
        del matrix[i][index]

# Exercise 2b

# Splits a graph into G(V, E↑) and G(V, E↓)
def split_graph(matrix):
    e_up = copy.deepcopy(matrix)
    e_down = copy.deepcopy(matrix)
    nodes = [Node(i, inf, None, i)
             for i in range(0, len(matrix))]
    
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            if nodes[i].importance < nodes[j].importance:
                e_down[i][j] = 0
            else:
                e_up[i][j] = 0

    return e_up, e_down

def bidirectional_dijkstra(matrix, src, dst):
    up, down = split_graph(matrix)
    dijkstra_up = dijkstra(up, src)
    dijkstra_down = backwards_dijkstra(down, dst)
    cost = dijkstra_up[0].distance + dijkstra_down[0].distance
    index = 0
    for i in range(1, len(dijkstra_up)):
        if dijkstra_up[i].distance + dijkstra_down[i].distance < cost:
            cost = dijkstra_up[i].distance + dijkstra_down[i].distance
            index = i
    print("Path between {} and {} passes through {} with total cost {}".format(src, dst, index, cost))


if __name__ == "__main__":
    # Example based on the weighted graph shown in the slides
    adj_matrix = [
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0],
        [0, 1, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 3],
        [3, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0]
    ]

    bidirectional_dijkstra(adj_matrix, 2, 0)
