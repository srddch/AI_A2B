from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def generate_route_graph_image(
    graph,
    route_result=None,
    output_path="static/routing_graph.png"
):
    """
    Generate a routing graph image for GUI display.

    Parameters:
        graph: graph dictionary loaded from graph_data.json
        route_result: result returned by find_route()
        output_path: image output path for Flask static display

    Returns:
        str: generated image path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_path = []

    if route_result and route_result.get("best_path"):
        best_path = route_result["best_path"]["nodes"]

    G = nx.DiGraph()

    for node in graph["nodes"]:
        G.add_node(str(node))

    for edge in graph["edges"]:
        from_node = str(edge["from"])
        to_node = str(edge["to"])
        distance = edge["features"]["distance_km"]

        G.add_edge(from_node, to_node, distance=distance)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 9))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=700,
        node_color="lightgray"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=14,
        width=1.2
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        font_weight="bold"
    )

    if best_path:
        best_edges = list(zip(best_path[:-1], best_path[1:]))

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=best_path,
            node_size=900,
            node_color="lightgreen"
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=best_edges,
            edge_color="green",
            arrows=True,
            arrowsize=18,
            width=3
        )

    plt.title("SCATS Routing Graph with Best Path", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return str(output_path)