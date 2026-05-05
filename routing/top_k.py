import heapq

from routing.edge_cost import get_edge_cost


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


def find_top_k_paths(graph, origin, destination, departure_time, model="lstm", k=5):
    """
    Find up to k candidate routes using an A*-style priority queue search.

    The search expands the currently lowest-cost candidate path first.
    Since the graph currently has no coordinates, the heuristic is set to 0.
    This makes the method behave like uniform-cost search while keeping the
    A* priority-queue structure.
    """
    origin = str(origin)
    destination = str(destination)

    if origin not in graph["nodes"] or destination not in graph["nodes"]:
        return {
            "paths": [],
            "best_path": None,
            "best_path_index": None
        }

    def heuristic(node):
        return 0

    results = []
    counter = 0

    # heap item:
    # (estimated_total_cost, counter, current_node, path, actual_cost_so_far)
    open_heap = []
    heapq.heappush(open_heap, (heuristic(origin), counter, origin, [origin], 0.0))

    while open_heap and len(results) < k:
        _, _, current_node, path, cost_so_far = heapq.heappop(open_heap)

        if current_node == destination:
            results.append({
                "nodes": path,
                "cost": round(cost_so_far, 2),
                "total_distance_km": calculate_path_distance(graph, path)
            })
            continue

        for edge in graph["edges"]:
            if str(edge["from"]) != str(current_node):
                continue

            next_node = str(edge["to"])

            # Avoid cycles such as A -> B -> A
            if next_node in path:
                continue

            edge_cost = get_edge_cost(edge, departure_time, model)
            new_cost = cost_so_far + edge_cost
            estimated_total_cost = new_cost + heuristic(next_node)

            counter += 1
            heapq.heappush(
                open_heap,
                (estimated_total_cost, counter, next_node, path + [next_node], new_cost)
            )

    return {
        "paths": results,
        "best_path": results[0] if results else None,
        "best_path_index": 0 if results else None
    }