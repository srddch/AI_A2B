import matplotlib.pyplot as plt
import networkx as nx


def generate_route_graph_image(
    graph,
    route_result,
    output_path="static/routing_graph.png"
):

    best_path = (
        route_result["best_path"]["nodes"]
        if route_result["best_path"]
        else []
    )

    G = nx.DiGraph()

    # Add nodes
    for node in graph["nodes"]:
        G.add_node(str(node))

    # Add edges
    for edge in graph["edges"]:

        from_node = str(edge["from"])
        to_node = str(edge["to"])

        distance = edge["features"]["distance_km"]

        G.add_edge(
            from_node,
            to_node,
            distance=distance
        )

    # Layout
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 9))

    # Normal nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=900,
        node_color="lightgray"
    )

    # Normal edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=18,
        width=1.5
    )

    # Labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold"
    )

    # Highlight best path
    if best_path:

        best_edges = list(
            zip(best_path[:-1], best_path[1:])
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=best_path,
            node_size=1000,
            node_color="lightgreen"
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=best_edges,
            edge_color="green",
            arrows=True,
            arrowsize=22,
            width=3
        )

    plt.title("Best Route Visualization")

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    plt.close()

    return output_path