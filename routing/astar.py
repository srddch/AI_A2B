from routing.top_k import find_top_k_paths

def find_route(graph, origin, destination, departure_time, model="A*"):
    result = find_top_k_paths(graph, origin, destination, departure_time)

    paths = result["paths"]

    return {
        "paths": paths,
        "best_path": paths[0] if paths else None
    }