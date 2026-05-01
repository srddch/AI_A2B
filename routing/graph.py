graph = {
    "A": [("B", 2), ("C", 4)],
    "B": [("D", 3)],
    "C": [("D", 1)],
    "D": []
}


def load_graph():
    return graph