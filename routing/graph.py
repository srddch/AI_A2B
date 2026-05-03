def load_graph():
    return {
        "nodes": ["A", "B", "C", "D"],
        "edges": [
            {"from": "A", "to": "B", "weight": 2, "features": {"distance": 2}},
            {"from": "A", "to": "C", "weight": 4, "features": {"distance": 4}},
            {"from": "B", "to": "D", "weight": 3, "features": {"distance": 3}},
            {"from": "C", "to": "D", "weight": 1, "features": {"distance": 1}},
        ]
    }