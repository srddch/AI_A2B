from routing.top_k import find_top_k_paths


def test_pipeline():
    res = find_top_k_paths("A", "D", "08:00")

    assert isinstance(res, dict)
    assert "paths" in res
    assert isinstance(res["paths"], list)