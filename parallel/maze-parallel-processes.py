from PIL import Image
import random
import numpy as np
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

# --- Maze generation parameters ---
GRAPH_SIZE = 50 # Reduced for demonstration speed
CELL_THICKNESS = 20
WALL_THICKNESS = 5

nodes = [(i,j) for j in range(GRAPH_SIZE) for i in range(GRAPH_SIZE)]
neighbors = lambda n : [(n[0]+dx,n[1]+dy) for dx,dy in ((-1,0),(1,0),(0,-1),(0,1))
                       if 0 <= n[0]+dx < GRAPH_SIZE and 0 <= n[1]+dy < GRAPH_SIZE]

class DisjointSet:
    def __init__(self, nodes):
        self.node_mapping = {val: self.DSNode(val, i) for i, val in enumerate(nodes)}

    def find_node(self, node):
        curr = self.node_mapping[node]
        if type(curr.parent) is int:
            return curr
        else:
            root = self.find_node(curr.parent.val)
            curr.parent = root
            return root

    def union(self, node1, node2):
        root1 = self.find_node(node1)
        root2 = self.find_node(node2)
        if root1 != root2:
            root1.parent = root2

    class DSNode:
        def __init__(self, val, parent):
            self.val = val
            self.parent = parent

def bfs_solve_parallel_procs(maze, start, end, nodes):
    t_start = time.perf_counter()

    graph = {node: [] for node in nodes}
    for edge in maze:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    with multiprocessing.Manager() as manager:
        # FIX: Using manager.dict() instead of manager.set()
        visited = manager.dict()
        visited[start] = True
        
        parent = manager.dict({start: None})
        queue = deque([start])
        found = False

        with ProcessPoolExecutor() as executor:
            while queue and not found:
                current = queue.popleft()
                
                # The worker function now uses the dict for 'visited'
                future = executor.submit(explore_step, current, graph, visited, parent, end)
                new_nodes, reached_end = future.result()
                
                for n in new_nodes:
                    queue.append(n)
                
                if reached_end:
                    found = True

        path = []
        path_dict = dict(parent) 
        if end not in path_dict: return None
        
        curr = end
        while curr is not None:
            path.append(curr)
            curr = path_dict.get(curr)
        path.reverse()

    t_end = time.perf_counter()
    print(f"Process-based solve time: {t_end - t_start:.4f} s")
    return path

def explore_step(current, graph, visited, parent, end):
    new_found = []
    found_end = False
    for neighbor in graph[current]:
        # Dictionary keys act like a set for 'in' lookups
        if neighbor not in visited:
            visited[neighbor] = True
            parent[neighbor] = current
            new_found.append(neighbor)
            if neighbor == end:
                found_end = True
    return new_found, found_end
            

if __name__ == "__main__":
    # --- Kruskal's Generation ---
    edges = [(node, nbor) for node in nodes for nbor in neighbors(node)]
    random.shuffle(edges)
    maze = []
    ds = DisjointSet(nodes)

    for edge in edges:
        if ds.find_node(edge[0]) != ds.find_node(edge[1]):
            ds.union(edge[0], edge[1])
            maze.append(edge)
        if len(maze) >= len(nodes) - 1:
            break

    # --- Solve ---
    start_node = (0, 0)
    end_node = (GRAPH_SIZE - 1, GRAPH_SIZE - 1)
    path = bfs_solve_parallel_procs(maze, start_node, end_node, nodes)

    # --- Image Rendering ---
    canvas_size = GRAPH_SIZE * (CELL_THICKNESS + WALL_THICKNESS) + WALL_THICKNESS
    img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Draw Maze
    for edge in maze:
        x1, y1 = edge[0]
        x2, y2 = edge[1]
        min_x = WALL_THICKNESS + min(x1, x2) * (CELL_THICKNESS + WALL_THICKNESS)
        max_x = WALL_THICKNESS + max(x1, x2) * (CELL_THICKNESS + WALL_THICKNESS)
        min_y = WALL_THICKNESS + min(y1, y2) * (CELL_THICKNESS + WALL_THICKNESS)
        max_y = WALL_THICKNESS + max(y1, y2) * (CELL_THICKNESS + WALL_THICKNESS)
        img[min_x:max_x + CELL_THICKNESS, min_y:max_y + CELL_THICKNESS] = 255

    # Draw Path
    if path:
        for cell in path:
            x, y = cell
            min_x = WALL_THICKNESS + x * (CELL_THICKNESS + WALL_THICKNESS)
            min_y = WALL_THICKNESS + y * (CELL_THICKNESS + WALL_THICKNESS)
            img[min_x:min_x + CELL_THICKNESS, min_y:min_y + CELL_THICKNESS] = 128

    im = Image.fromarray(img)
    im.save("solved_maze_multiprocessing.png")
    im.show()
