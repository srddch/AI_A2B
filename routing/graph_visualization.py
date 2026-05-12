import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


def generate_route_graph_image(
    graph,
    route_result,
    output_path="static/routing_graph.png"
):

    # =========================
    # Get best path
    # =========================

    best_path = []

    if route_result and route_result.get("best_path"):

        best_path = route_result["best_path"]["nodes"]

    # =========================
    # Build graph
    # =========================

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

    # =========================
    # Layout
    # =========================

    pos = nx.spring_layout(
        G,
        seed=42
    )

    # =========================
    # Create figure
    # =========================

    fig = plt.figure(figsize=(14, 9))

    # =========================
    # Draw normal nodes
    # =========================

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=900,
        node_color="lightgray"
    )

    # =========================
    # Draw normal edges
    # =========================

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=18,
        width=1.5
    )

    # =========================
    # Draw labels
    # =========================

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold"
    )

    # =========================
    # Highlight best path
    # =========================

    if best_path:

        best_edges = list(
            zip(
                best_path[:-1],
                best_path[1:]
            )
        )

        # Highlight nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=best_path,
            node_size=1000,
            node_color="lightgreen"
        )

        # Highlight edges
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=best_edges,
            edge_color="green",
            arrows=True,
            arrowsize=22,
            width=3
        )

    # =========================
    # Edge labels
    # =========================

    edge_labels = {

        (u, v): f"{d['distance']:.2f} km"

        for u, v, d in G.edges(data=True)

    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8
    )

    # =========================
    # Title
    # =========================

    plt.title(
        "Best Route Visualization",
        fontsize=16
    )

    plt.axis("off")

    plt.tight_layout()

    # =========================
    # Save image
    # =========================

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # =========================
    # Return image path
    # =========================

    return str(output_path)