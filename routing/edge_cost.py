from traffic.predictor import predict_traffic
from traffic.travel_time import flow_to_travel_time

def get_edge_cost(from_node, to_node, distance, time):
    flow = predict_traffic(to_node, time)
    return flow_to_travel_time(flow, distance)