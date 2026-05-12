from traffic.predictor import predict_travel_time_by_departure

_edge_cost_cache = {}

def get_edge_cost(edge, departure_time, model="lstm"):

    """
    Convert a graph edge into a traffic-based travel time cost.

    The routing layer prepares:
        from_site
        to_site
        distance_km
        departure_time

    The traffic layer then predicts the travel time using the selected
    traffic model, such as "lstm" or "gru".

    Returns:
        travel time in minutes
    """
    

    cache_key = (
        str(edge["from"]),
        str(edge["to"]),
        float(edge["features"]["distance_km"]),
        str(departure_time),
        str(model).lower()
    )

    if cache_key in _edge_cost_cache:
        return _edge_cost_cache[cache_key]
    


    edge_features = {
        "from_site": int(edge["from"]),
        "to_site": int(edge["to"]),
        "distance_km": float(edge["features"]["distance_km"]),
        "departure_time": departure_time
    }

    travel_time = predict_travel_time_by_departure(edge_features, model_name=model)
    _edge_cost_cache[cache_key] = travel_time

    return travel_time



def get_path_edge_details(graph, path, departure_time, model="lstm"):
    """
    Build edge-level details for a completed route.

    Each edge detail includes:
        - static distance_km
        - dynamic predicted travel_time_min
    """
    edge_details = []

    for i in range(len(path) - 1):
        from_node = str(path[i])
        to_node = str(path[i + 1])

        matching_edge = None

        for edge in graph["edges"]:
            if str(edge["from"]) == from_node and str(edge["to"]) == to_node:
                matching_edge = edge
                break

        if matching_edge is None:
            continue

        travel_time = get_edge_cost(matching_edge, departure_time, model)

        edge_details.append({
            "from": from_node,
            "to": to_node,
            "distance_km": float(matching_edge["features"]["distance_km"]),
            "travel_time_min": round(travel_time, 2)
        })

    return edge_details