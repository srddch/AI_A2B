from traffic.predictor import predict_travel_time_by_departure


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
    """
    Calculate the traffic-based travel time for one edge.

    edge format:
    {
        "from": "2000",
        "to": "4049",
        "features": {
            "distance_km": 0.65
        }
    }
    """

    edge_features = {
        "from_site": int(edge["from"]),
        "to_site": int(edge["to"]),
        "distance_km": float(edge["features"]["distance_km"]),
        "departure_time": departure_time
    }

    return predict_travel_time_by_departure(edge_features, model_name=model)