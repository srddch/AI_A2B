from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routing.graph import load_graph
from routing.astar import find_route


def test_valid_routes():
    graph = load_graph()

    test_cases = [
        ("2000", "3002"),
        ("2000", "4035"),
        ("3682", "3002"),
        ("4040", "3002"),
    ]

    for origin, destination in test_cases:
        result = find_route(
            graph=graph,
            origin=origin,
            destination=destination,
            departure_time="2006-10-01 08:00:00",
            model="lstm"
        )

        assert len(result["paths"]) > 0
        assert len(result["paths"]) <= 5
        assert result["best_path"] == result["paths"][0]
        assert result["best_path_index"] == 0
        assert result.get("astar_best_path") is not None
        assert result.get("astar_best_path")["nodes"] == result["best_path"]["nodes"]

        costs = [path["cost"] for path in result["paths"]]
        assert costs == sorted(costs)


def test_invalid_or_no_path_routes():
    graph = load_graph()

    test_cases = [
        ("9999", "3002"),
        ("3002", "2000"),
    ]

    for origin, destination in test_cases:
        result = find_route(
            graph=graph,
            origin=origin,
            destination=destination,
            departure_time="2006-10-01 08:00:00",
            model="lstm"
        )

        assert result["paths"] == []
        assert result["best_path"] is None
        assert result["best_path_index"] is None
        assert result.get("astar_best_path") is None


def test_invalid_model_fallback():
    graph = load_graph()

    result = find_route(
        graph=graph,
        origin="2000",
        destination="3002",
        departure_time="2006-10-01 08:00:00",
        model="invalid_model"
    )

    assert len(result["paths"]) > 0
    assert result["best_path"] == result["paths"][0]
    assert result["best_path_index"] == 0



if __name__ == "__main__":
    test_valid_routes()
    test_invalid_or_no_path_routes()
    test_invalid_model_fallback()
    print("All routing tests passed.")