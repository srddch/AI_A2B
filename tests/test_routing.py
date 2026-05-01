from routing.top_k import find_top_k_paths


def test_basic():
    res = find_top_k_paths("A", "D", "08:00")
    assert "paths" in res
    assert len(res["paths"]) > 0


def test_best_path():
    res = find_top_k_paths("A", "D", "08:00")
    assert res["best_path_index"] == 0