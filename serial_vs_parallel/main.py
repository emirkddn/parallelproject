import time
import random
import numpy as np
from PIL import Image
from collections import deque
from concurrent.futures import ThreadPoolExecutor

GRAPH_SIZE = 1000
CELL_THICKNESS = 10
WALL_THICKNESS = 2

class DisjointSet:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1
            return True
        return False

def generate_maze():
    nodes = [(i, j) for j in range(GRAPH_SIZE) for i in range(GRAPH_SIZE)]
    adj = lambda n: [(n[0]+dx, n[1]+dy) for dx, dy in ((-1,0),(1,0),(0,-1),(0,1))
                     if 0 <= n[0]+dx < GRAPH_SIZE and 0 <= n[1]+dy < GRAPH_SIZE]
    edges = []
    for n in nodes:
        for nbor in adj(n):
            if n < nbor: edges.append((n, nbor))
    random.shuffle(edges)
    ds = DisjointSet(nodes)
    maze_edges = []
    for u, v in edges:
        if ds.union(u, v):
            maze_edges.append((u, v))
    return maze_edges, nodes

def expand_chunk(chunk, graph, visited):
    local_results = []
    for node in chunk:
        for nbor in graph[node]:
            if nbor not in visited:
                local_results.append((nbor, node))
    return local_results

def solve_serial(graph, start, end):
    t0 = time.perf_counter()
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue:
        curr = queue.popleft()
        if curr == end: break
        for nbor in graph[curr]:
            if nbor not in visited:
                visited.add(nbor)
                parent[nbor] = curr
                queue.append(nbor)
    
    t1 = time.perf_counter()
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = parent.get(curr)
    return path[::-1], (t1 - t0)

def solve_multithread(graph, start, end, worker_num):
    t0 = time.perf_counter()
    frontier = [start]
    visited = {start}
    parent_map = {start: None}
    found = False
    workers = worker_num

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while frontier and not found:

            size = max(1, len(frontier) // workers)
            chunks = [frontier[i:i + size] for i in range(0, len(frontier), size)]
            
            results = list(executor.map(expand_chunk, chunks, [graph]*len(chunks), [visited]*len(chunks)))
            
            new_frontier = []
            for batch in results:
                for nbor, p in batch:
                    if nbor not in visited:
                        visited.add(nbor)
                        parent_map[nbor] = p
                        new_frontier.append(nbor)
                        if nbor == end: found = True
            frontier = new_frontier

    t1 = time.perf_counter()
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = parent_map.get(curr)
    return path[::-1], (t1 - t0)

if __name__ == "__main__":
    print(f"--- Maze Generation ({GRAPH_SIZE}x{GRAPH_SIZE}) ---")
    g_start = time.perf_counter()
    maze_edges, all_nodes = generate_maze()
    print(f"Generation Time: {time.perf_counter() - g_start:.4f} s")

    graph = {n: [] for n in all_nodes}
    for u, v in maze_edges:
        graph[u].append(v)
        graph[v].append(u)

    start_node, end_node = (0, 0), (GRAPH_SIZE - 1, GRAPH_SIZE - 1)

    path_s, time_s = solve_serial(graph, start_node, end_node)
    print(f"Serial Solve Time: {time_s:.4f} s")

    path_t, time_t = solve_multithread(graph, start_node, end_node, 2)
    print(f"Thread Solve Time (2 workers): {time_t:.4f} s")

    path_t, time_t = solve_multithread(graph, start_node, end_node, 4)
    print(f"Thread Solve Time (4 workers): {time_t:.4f} s")

    path_t, time_t = solve_multithread(graph, start_node, end_node, 8)
    print(f"Thread Solve Time (8 workers): {time_t:.4f} s")

    path_t, time_t = solve_multithread(graph, start_node, end_node, 16)
    print(f"Thread Solve Time (16 workers): {time_t:.4f} s")

    print("Generating Image...")
    dim = GRAPH_SIZE * (CELL_THICKNESS + WALL_THICKNESS) + WALL_THICKNESS
    img = np.zeros((dim, dim), dtype=np.uint8)

    for u, v in maze_edges:
        x0 = WALL_THICKNESS + min(u[0], v[0]) * (CELL_THICKNESS + WALL_THICKNESS)
        y0 = WALL_THICKNESS + min(u[1], v[1]) * (CELL_THICKNESS + WALL_THICKNESS)

        x1 = x0 + (CELL_THICKNESS if u[0] == v[0] else 2*CELL_THICKNESS + WALL_THICKNESS)
        y1 = y0 + (CELL_THICKNESS if u[1] == v[1] else 2*CELL_THICKNESS + WALL_THICKNESS)
        img[x0:x1, y0:y1] = 255

    for cell in path_s:
        x = WALL_THICKNESS + cell[0] * (CELL_THICKNESS + WALL_THICKNESS)
        y = WALL_THICKNESS + cell[1] * (CELL_THICKNESS + WALL_THICKNESS)
        img[x:x+CELL_THICKNESS, y:y+CELL_THICKNESS] = 150

    Image.fromarray(img).save("maze_solve_comparison.png")
    print("Done!")