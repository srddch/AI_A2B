import json
from pathlib import Path


def load_graph():

    """
    Load the road network graph from data/processed/graph_data.json.

    Graph format:
        {
            "nodes": ["2000", "3002", ...],
            "edges": [
                {
                    "from": "2000",
                    "to": "3682",
                    "weight": 1.655,
                    "features": {
                        "distance_km": 1.655
                    }
                }
            ]
        }

    Returns:
        graph dictionary used by the routing module
    """
    base_dir = Path(__file__).resolve().parent.parent
    graph_path = base_dir / "data" / "processed" / "graph_data.json"

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    return graph
