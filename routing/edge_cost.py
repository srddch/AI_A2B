from traffic.predictor import predict_travel_time


def get_edge_cost(edge, time):
    return predict_travel_time(edge["features"], model_name="lstm")