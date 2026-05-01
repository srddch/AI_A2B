from routing.graph import graph
from routing.edge_cost import get_edge_cost


def dfs_all_paths(start, end, path=None):
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return [path]

    paths = []

    for (node, _) in graph.get(start, []):
        if node not in path:
            new_paths = dfs_all_paths(node, end, path)
            paths.extend(new_paths)

    return paths


def calculate_path_cost(path, time):
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
        cost = calculate_path_cost(p, time)
        results.append({
            "nodes": p,
            "cost": cost
        })

    results.sort(key=lambda x: x["cost"])

    return {
        "paths": results[:k],
        "best_path_index": 0
    }