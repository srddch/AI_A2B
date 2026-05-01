from routing.top_k import find_top_k_paths

def test_path():
    routes = find_top_k_paths("A", "D", "08:00")
    assert len(routes) > 0