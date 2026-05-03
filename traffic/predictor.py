import random


def predict_travel_time(edge_features, model_name="lstm"):
    distance = edge_features.get("distance", 1)

    # 假流量
    flow = random.randint(50, 200)

    speed = 60
    base_time = distance / speed * 3600

    congestion = flow / 100
    delay = 30

    return base_time * (1 + congestion) + delay