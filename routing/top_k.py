from routing.edge_cost import get_edge_cost


def dfs_all_paths(graph, start, end, path=None):
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return [path]

    paths = []

    for edge in graph["edges"]:
        if edge["from"] == start:
            node = edge["to"]
            if node not in path:
                new_paths = dfs_all_paths(graph, node, end, path)
                paths.extend(new_paths)

    return paths


def calculate_path_cost(graph, path, time):
    total = 0

    for i in range(len(path) - 1):
        for edge in graph["edges"]:
            if edge["from"] == path[i] and edge["to"] == path[i + 1]:
                total += get_edge_cost(edge, time)

    return total


def find_top_k_paths(graph, origin, destination, time, k=5):
    paths = dfs_all_paths(graph, origin, destination)

    results = []

    for p in paths:
        cost = calculate_path_cost(graph, p, time)
        results.append({
            "nodes": p,
            "cost": cost
        })

    results.sort(key=lambda x: x["cost"])

    return {
        "paths": results[:k]
    }