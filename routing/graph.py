import json
from pathlib import Path


def load_graph():
    """
    Load the road network graph from data/processed/full_graph_data.json.

    The graph uses bidirectional directed edge pairs for nearby SCATS sites.
    """
    base_dir = Path(__file__).resolve().parent.parent
    graph_path = base_dir / "data" / "processed" / "full_graph_data.json"

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    return graph