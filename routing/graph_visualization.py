from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def generate_route_graph_image(
    graph,
    route_result=None,
    output_path="static/routing_graph.png",
    mode="top_k"
):
    """
    Generate a routing graph image for GUI display.

    Modes:
        - "top_k": show only the nodes and edges used by the returned top-k routes
        - "best": show only the best path
        - "full": show the complete SCATS graph

    Returns:
        str: generated image path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_path = []

    if route_result and route_result.get("best_path"):
        best_path = route_result["best_path"]["nodes"]

    paths = []

    if route_result and route_result.get("paths"):
        paths = route_result["paths"]

    G = nx.DiGraph()

    # Decide which nodes and edges should be shown.
    if mode == "best" and best_path:
        nodes_to_draw = set(best_path)
        edges_to_draw = set(zip(best_path[:-1], best_path[1:]))

    elif mode == "top_k" and paths:
        nodes_to_draw = set()
        edges_to_draw = set()

        for route in paths:
            route_nodes = route["nodes"]
            nodes_to_draw.update(route_nodes)
            edges_to_draw.update(zip(route_nodes[:-1], route_nodes[1:]))

    else:
        nodes_to_draw = set(str(node) for node in graph["nodes"])
        edges_to_draw = set(
            (str(edge["from"]), str(edge["to"]))
            for edge in graph["edges"]
        )

    # Add selected nodes.
    for node in nodes_to_draw:
        G.add_node(str(node))

    # Add selected edges only.
    for edge in graph["edges"]:
        from_node = str(edge["from"])
        to_node = str(edge["to"])

        if (from_node, to_node) in edges_to_draw:
            distance = edge["features"]["distance_km"]
            G.add_edge(from_node, to_node, distance=distance)

    pos = nx.spring_layout(G, seed=42, k=1.4)

    plt.figure(figsize=(14, 9))

    # Draw candidate route nodes.
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=750,
        node_color="lightgray"
    )

    # Draw all candidate route edges in gray.
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(G.edges()),
        edge_color="gray",
        arrows=True,
        arrowsize=14,
        width=1.4
    )

    # Draw labels.
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold"
    )

    # Highlight best path.
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

        # Only label best path edges to avoid clutter.
        best_edge_labels = {}

        if route_result and route_result.get("best_path"):
            for edge_detail in route_result["best_path"].get("edge_details", []):
                from_node = str(edge_detail["from"])
                to_node = str(edge_detail["to"])
                distance = edge_detail["distance_km"]
                travel_time = edge_detail["travel_time_min"]

                best_edge_labels[(from_node, to_node)] = (
                    f"{distance:.2f} km\n{travel_time:.2f} min"
                )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=best_edge_labels,
            font_size=8,
            label_pos=0.5
        )

    if mode == "top_k":
        title = "Top-k Routes Subgraph with Best Path Highlighted"
    elif mode == "best":
        title = "Best Path"
    else:
        title = "Full SCATS Routing Graph"

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return str(output_path)