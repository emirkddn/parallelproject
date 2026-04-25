## Kruskal's Algorithm for Maze Generation
## Neil Thistlethwaite

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import random
import numpy as np
from collections import deque
import time
from threading import Lock


global GRAPH_SIZE, CELL_THICKNESS, WALL_THICKNESS

GRAPH_SIZE = 400
CELL_THICKNESS = 20
WALL_THICKNESS = 5

general_start = time.perf_counter()

nodes = [(i,j) for j in range(GRAPH_SIZE) for i in range(GRAPH_SIZE)]
neighbors = lambda n : [(n[0]+dx,n[1]+dy) for dx,dy in ((-1,0),(1,0),(0,-1),(0,1))
                       if n[0]+dx >= 0 and n[0]+dx < GRAPH_SIZE and n[1]+dy >= 0 and n[1]+dy < GRAPH_SIZE]

class DisjointSet:
    def __init__(self, nodes):
        self.node_mapping = {}
        for i,val in enumerate(nodes):
            n = self.DSNode(val, i)
            self.node_mapping[val] = n

    def find(self, node):
        return self.find_node(node).parent

    def find_node(self, node):
        if type(self.node_mapping[node].parent) is int:
            return self.node_mapping[node]
        else:
            parent_node = self.find_node(self.node_mapping[node].parent.val)
            self.node_mapping[node].parent = parent_node
            return parent_node

    def union(self, node1, node2):
        parent1 = self.find_node(node1)
        parent2 = self.find_node(node2)
        if parent1.parent != parent2.parent:
            parent1.parent = parent2

    class DSNode:
        def __init__(self, val, parent):
            self.val = val
            self.parent = parent

def bfs_solve(maze, start, end, nodes):
    
    t_start = time.perf_counter()
    
    graph = {node: [] for node in nodes}
    for edge in maze:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    
    queue = deque([start])
    visited = set([start])
    parent = {start: None}
    found = False
    
    while queue:
        current = queue.popleft()
        if current == end:
            found = True
            break
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    if not found:
        return None
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    print(f"Serial solve time: {elapsed:.4f} s")

    return path

def bfs_solve_parallel(maze, start, end, nodes):

    t_start = time.perf_counter()

    graph = {node: [] for node in nodes}
    for edge in maze:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    
    visited_lock = Lock()
    parent_lock = Lock()
    visited = set([start])
    parent = {start: None}
    queue = deque([start])
    found = [False]
    
    def explore_neighbors(current):
        if found[0]:
            return
        for neighbor in graph[current]:
            with visited_lock:
                if neighbor not in visited:
                    visited.add(neighbor)
                    with parent_lock:
                        parent[neighbor] = current
                    queue.append(neighbor)
                    if neighbor == end:
                        found[0] = True
                        return
    
    while queue and not found[0]:
        current = queue.popleft()
        if current == end:
            found[0] = True
            break
        with ThreadPoolExecutor(max_workers=32) as executor:
            executor.submit(explore_neighbors, current)
    
    if not found[0]:
        return None
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    t_end = time.perf_counter()
    elapsed = (t_end - t_start)
    print(f"Parallel solve time: {elapsed:.4f} s")

    return path

edges = [(node, nbor) for node in nodes for nbor in neighbors(node)]
maze = []
ds = DisjointSet(nodes)

while len(maze) < len(nodes)-1:
    edge = edges.pop(random.randint(0, len(edges)-1))
    if ds.find(edge[0]) != ds.find(edge[1]):
        ds.union(edge[0], edge[1])
        maze.append(edge)

img = np.zeros((GRAPH_SIZE * (CELL_THICKNESS + WALL_THICKNESS) + WALL_THICKNESS,
                GRAPH_SIZE * (CELL_THICKNESS + WALL_THICKNESS) + WALL_THICKNESS),dtype=np.uint8)

for edge in maze:
    min_x = WALL_THICKNESS+min(edge[0][0],edge[1][0])*(CELL_THICKNESS + WALL_THICKNESS)
    max_x = WALL_THICKNESS+max(edge[0][0],edge[1][0])*(CELL_THICKNESS + WALL_THICKNESS)
    min_y = WALL_THICKNESS+min(edge[0][1],edge[1][1])*(CELL_THICKNESS + WALL_THICKNESS)
    max_y = WALL_THICKNESS+max(edge[0][1],edge[1][1])*(CELL_THICKNESS + WALL_THICKNESS)
    img[min_x:max_x+CELL_THICKNESS,min_y:max_y+CELL_THICKNESS] = 255

start = (0, 0)
end = (GRAPH_SIZE - 1, GRAPH_SIZE - 1)
path_serial = bfs_solve(maze, start, end, nodes)
path_parallel = bfs_solve_parallel(maze, start, end, nodes)
general_end = time.perf_counter()

time_elapsed = general_end - general_start
print(f"General time elapsed: {time_elapsed:.4f} s")

if path_serial:
    for cell in path_serial:
        min_x = WALL_THICKNESS + cell[0] * (CELL_THICKNESS + WALL_THICKNESS)
        max_x = min_x + CELL_THICKNESS
        min_y = WALL_THICKNESS + cell[1] * (CELL_THICKNESS + WALL_THICKNESS)
        max_y = min_y + CELL_THICKNESS
        img[min_x:max_x, min_y:max_y] = 128

im = Image.fromarray(img)
im.show()
