import heapq

def calculate_path_distance(graph, path):
    total_distance = 0.0

    for i in range(len(path) - 1):
        from_node = str(path[i])
        to_node = str(path[i + 1])

        for edge in graph["edges"]:
            if str(edge["from"]) == from_node and str(edge["to"]) == to_node:
                total_distance += float(edge["features"]["distance_km"])
                break

    return round(total_distance, 3)

from routing.top_k import find_top_k_paths
from routing.edge_cost import get_edge_cost

def heuristic(node, destination, graph):
    """
    A* heuristic function.

    Current minimum version returns 0 because graph_data.json does not yet
    contain coordinates. This makes A* behave like uniform-cost search while
    keeping the A* structure.
    """
    return 0


def get_outgoing_edges(graph, node):
    """
    Get all directed outgoing edges from a node.
    """
    return [
        edge for edge in graph["edges"]
        if str(edge["from"]) == str(node)
    ]

def astar_search(graph, origin, destination, departure_time, model="lstm"):
    """
    Find the single best path using A* search.

    Edge cost is calculated by the traffic prediction layer.
    """
    origin = str(origin)
    destination = str(destination)

    if origin not in graph["nodes"] or destination not in graph["nodes"]:
        return None

    counter = 0
    open_heap = []

    start_g = 0.0
    start_h = heuristic(origin, destination, graph)
    start_f = start_g + start_h

    heapq.heappush(open_heap, (start_f, counter, origin, [origin], start_g))

    visited = set()

    while open_heap:
        current_f, _, current_node, path, current_g = heapq.heappop(open_heap)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == destination:
            return {
                "nodes": path,
                "cost": round(current_g, 2),
                "total_distance_km": calculate_path_distance(graph, path)
            }

        for edge in get_outgoing_edges(graph, current_node):
            next_node = str(edge["to"])

            if next_node in visited:
                continue

            edge_cost = get_edge_cost(edge, departure_time, model)
            new_g = current_g + edge_cost
            new_h = heuristic(next_node, destination, graph)
            new_f = new_g + new_h

            counter += 1
            heapq.heappush(
                open_heap,
                (new_f, counter, next_node, path + [next_node], new_g)
            )

    return None

def find_route(graph, origin, destination, departure_time, model="lstm"):
    """
    Unified routing entry point used by main.py and GUI.

    Parameters:
        graph: road network graph loaded from graph_data.json
        origin: origin SCATS site number as string or int
        destination: destination SCATS site number as string or int
        departure_time: selected departure time
        model: traffic prediction model name, either "lstm" or "gru"

    Returns:
        dict with:
            paths: up to 5 paths sorted by cost ascending
            best_path: the first path in paths, or None if no path exists
            best_path_index: 0 if a path exists, otherwise None
    """

    model = str(model).lower()

    if model not in ["lstm", "gru"]:
        model = "lstm"

    astar_best_path = astar_search(graph, origin, destination, departure_time, model=model)
    result = find_top_k_paths(graph, origin, destination, departure_time, model=model)

    paths = result["paths"]

    return {
    "paths": paths,
    "best_path": result.get("best_path"),
    "best_path_index": result.get("best_path_index"),
    "astar_best_path": astar_best_path
}