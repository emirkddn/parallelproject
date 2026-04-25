## Kruskal's Algorithm for Maze Generation
## Neil Thistlethwaite

from PIL import Image
import random
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

global GRAPH_SIZE, CELL_THICKNESS, WALL_THICKNESS

## Maze generation parameters. Change as desired.
GRAPH_SIZE = 200
CELL_THICKNESS = 20
WALL_THICKNESS = 5

nodes = [(i,j) for j in range(GRAPH_SIZE) for i in range(GRAPH_SIZE)]
neighbors = lambda n : [(n[0]+dx,n[1]+dy) for dx,dy in ((-1,0),(1,0),(0,-1),(0,1))
                       if n[0]+dx >= 0 and n[0]+dx < GRAPH_SIZE and n[1]+dy >= 0 and n[1]+dy < GRAPH_SIZE]

## Somewhat naive implementation, as it doesn't do rank balancing,
## but this could easily be replaced with something more efficient.
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

def bfs_solve_parallel(maze, start, end, nodes):
    # Create graph from maze edges

    t_start = time.perf_counter()

    graph = {node: [] for node in nodes}
    for edge in maze:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    
    # BFS (simulated parallel with threads for exploration)
    from threading import Lock
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
        # Parallel exploration of neighbors
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.submit(explore_neighbors, current)
    
    if not found[0]:
        return None
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    t_end = time.perf_counter()
    elapsed = (t_end - t_start)
    print(f"Parallel solve time: {elapsed} s")

    return path

## Kruskal's Algorithm
edges = [(node, nbor) for node in nodes for nbor in neighbors(node)]
maze = []
ds = DisjointSet(nodes)

while len(maze) < len(nodes)-1:
    edge = edges.pop(random.randint(0, len(edges)-1))
    if ds.find(edge[0]) != ds.find(edge[1]):
        ds.union(edge[0], edge[1])
        maze.append(edge)

## Now convert to an image
img = np.zeros((GRAPH_SIZE * (CELL_THICKNESS + WALL_THICKNESS) + WALL_THICKNESS,
                GRAPH_SIZE * (CELL_THICKNESS + WALL_THICKNESS) + WALL_THICKNESS),dtype=np.uint8)

for edge in maze:
    min_x = WALL_THICKNESS+min(edge[0][0],edge[1][0])*(CELL_THICKNESS + WALL_THICKNESS)
    max_x = WALL_THICKNESS+max(edge[0][0],edge[1][0])*(CELL_THICKNESS + WALL_THICKNESS)
    min_y = WALL_THICKNESS+min(edge[0][1],edge[1][1])*(CELL_THICKNESS + WALL_THICKNESS)
    max_y = WALL_THICKNESS+max(edge[0][1],edge[1][1])*(CELL_THICKNESS + WALL_THICKNESS)
    img[min_x:max_x+CELL_THICKNESS,min_y:max_y+CELL_THICKNESS] = 255

# Solve the maze using parallel BFS
start = (0, 0)
end = (GRAPH_SIZE - 1, GRAPH_SIZE - 1)
path = bfs_solve_parallel(maze, start, end, nodes)

# Draw the path in gray (128)
if path:
    for cell in path:
        min_x = WALL_THICKNESS + cell[0] * (CELL_THICKNESS + WALL_THICKNESS)
        max_x = min_x + CELL_THICKNESS
        min_y = WALL_THICKNESS + cell[1] * (CELL_THICKNESS + WALL_THICKNESS)
        max_y = min_y + CELL_THICKNESS
        img[min_x:max_x, min_y:max_y] = 128

im = Image.fromarray(img)
im.show()

## Save solved maze
im.save("solved_maze_parallel.png")
print("Solved maze saved as solved_maze_parallel.png")