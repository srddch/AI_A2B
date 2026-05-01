def flow_to_travel_time(flow, distance):
    speed = 60
    base_time = distance / speed * 3600

    congestion = flow / 100
    delay = 30

    return base_time * (1 + congestion) + delay