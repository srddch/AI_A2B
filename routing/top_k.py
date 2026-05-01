from routing.graph import graph
from routing.edge_cost import get_edge_cost

def dfs_all_paths(start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]

    paths = []
    for (node, _) in graph.get(start, []):
        if node not in path:
            newpaths = dfs_all_paths(node, end, path)
            paths.extend(newpaths)
    return paths


def calculate_path_time(path, time):
    total = 0
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]

        for (neighbor, dist) in graph[from_node]:
            if neighbor == to_node:
                total += get_edge_cost(from_node, to_node, dist, time)

    return total


def find_top_k_paths(origin, destination, time, k=5):
    paths = dfs_all_paths(origin, destination)

    results = []
    for p in paths:
        t = calculate_path_time(p, time)
        results.append({"path": p, "time": t})

    results.sort(key=lambda x: x["time"])
    return results[:k]