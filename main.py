from ui.gui import run_app
from routing.top_k import find_top_k_paths
from routing.graph import load_graph


def create_router(graph):
    def router(origin, destination, time, model=None):
        return find_top_k_paths(origin, destination, time)

    return router


if __name__ == "__main__":
    graph = load_graph()
    run_app(create_router(graph))