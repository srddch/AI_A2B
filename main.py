from ui.gui import run_app
from routing.astar import find_route
from routing.graph import load_graph


if __name__ == "__main__":
    graph = load_graph()
    run_app(lambda o, d, t, m: find_route(graph, o, d, t, m))