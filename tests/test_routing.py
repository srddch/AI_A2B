from routing.astar import find_route
from routing.graph import load_graph


def test_basic():
    graph = load_graph()
    res = find_route(graph, "A", "D", "08:00")

    assert "paths" in res
    assert "best_path" in res
    assert len(res["paths"]) > 0