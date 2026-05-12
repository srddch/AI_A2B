from pathlib import Path
import csv

import matplotlib.pyplot as plt
import networkx as nx


def load_coordinate_positions(graph):
    """
    Load node positions from site_metadata_merged.csv.

    Position format for networkx:
        node_id -> (longitude, latitude)

    If coordinates are not available, the missing nodes will later fall back
    to spring_layout.
    """
    base_dir = Path(__file__).resolve().parent.parent
    metadata_path = base_dir / "data" / "processed" / "site_metadata_merged.csv"

    positions = {}

    if not metadata_path.exists():
        return positions

    with open(metadata_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                site_number = str(int(float(row["site_number"])))
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])

                if latitude != 0 and longitude != 0:
                    positions[site_number] = (longitude, latitude)

            except (KeyError, ValueError, TypeError):
                continue

    return {
        str(node): positions[str(node)]
        for node in graph["nodes"]
        if str(node) in positions
    }


def generate_route_graph_image(
    graph,
    route_result=None,
    output_path="static/routing_graph.png",
    mode="full_highlight"
):
    """
    Generate a routing graph image for GUI display.

    Modes:
        - "full": show the complete SCATS graph in grey
        - "full_highlight": show the complete SCATS graph and highlight best path
        - "top_k": show only top-k route subgraph and highlight best path
        - "best": show only the best path

    Returns:
        str: generated image path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_path = []

    if route_result and route_result.get("best_path"):
        best_path = [str(node) for node in route_result["best_path"]["nodes"]]

    paths = []

    if route_result and route_result.get("paths"):
        paths = route_result["paths"]

    G = nx.DiGraph()

    if mode == "best" and best_path:
        nodes_to_draw = set(best_path)
        edges_to_draw = set(zip(best_path[:-1], best_path[1:]))

    elif mode == "top_k" and paths:
        nodes_to_draw = set()
        edges_to_draw = set()

        for route in paths:
            route_nodes = [str(node) for node in route["nodes"]]
            nodes_to_draw.update(route_nodes)
            edges_to_draw.update(zip(route_nodes[:-1], route_nodes[1:]))

    else:
        nodes_to_draw = set(str(node) for node in graph["nodes"])
        edges_to_draw = set(
            (str(edge["from"]), str(edge["to"]))
            for edge in graph["edges"]
        )

    for node in nodes_to_draw:
        G.add_node(str(node))

    for edge in graph["edges"]:
        from_node = str(edge["from"])
        to_node = str(edge["to"])

        if (from_node, to_node) in edges_to_draw:
            distance = float(edge["features"]["distance_km"])
            G.add_edge(from_node, to_node, distance=distance)

    # Use real coordinate positions where possible.
    pos = load_coordinate_positions(graph)

    # Fallback for missing node positions.
    missing_nodes = [node for node in G.nodes() if node not in pos]
    if missing_nodes:
        fallback_pos = nx.spring_layout(G, seed=42, k=1.4)
        for node in missing_nodes:
            pos[node] = fallback_pos[node]

    plt.figure(figsize=(15, 10))

    # Draw full / selected graph background.
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=360,
        node_color="lightgray",
        edgecolors="gray",
        linewidths=0.8
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(G.edges()),
        edge_color="lightgray",
        arrows=True,
        arrowsize=10,
        width=1.0,
        alpha=0.7
    )

    # Draw general node labels.
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=7,
        font_weight="bold"
    )

    highlight_enabled = mode in ["full_highlight", "top_k", "best"]

    if highlight_enabled and best_path:
        best_edges = list(zip(best_path[:-1], best_path[1:]))

        # Highlight best path edges.
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=best_edges,
            edge_color="green",
            arrows=True,
            arrowsize=18,
            width=3.2
        )

        # Highlight best path nodes.
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=best_path,
            node_size=620,
            node_color="lightgreen",
            edgecolors="green",
            linewidths=1.5
        )

        # Mark origin and destination more clearly.
        origin = best_path[0]
        destination = best_path[-1]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[origin],
            node_size=780,
            node_color="orange",
            edgecolors="black",
            linewidths=1.5
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[destination],
            node_size=780,
            node_color="red",
            edgecolors="black",
            linewidths=1.5
        )

        # Label best path nodes with order.
        path_order_labels = {}

        for index, node in enumerate(best_path, start=1):
            if node == origin:
                path_order_labels[node] = f"START\n{index}: {node}"
            elif node == destination:
                path_order_labels[node] = f"END\n{index}: {node}"
            else:
                path_order_labels[node] = f"{index}: {node}"

        nx.draw_networkx_labels(
            G,
            pos,
            labels=path_order_labels,
            font_size=8,
            font_weight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "boxstyle": "round,pad=0.2"
            }
        )

        # Only label best path edges to avoid clutter.
        best_edge_labels = {}

        if route_result and route_result.get("best_path"):
            for edge_detail in route_result["best_path"].get("edge_details", []):
                from_node = str(edge_detail["from"])
                to_node = str(edge_detail["to"])
                distance = float(edge_detail["distance_km"])
                travel_time = float(edge_detail["travel_time_min"])

                best_edge_labels[(from_node, to_node)] = (
                    f"{distance:.2f} km\n{travel_time:.2f} min"
                )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=best_edge_labels,
            font_size=7,
            label_pos=0.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "boxstyle": "round,pad=0.15"
            }
        )

    if mode == "full":
        title = "Full SCATS Routing Graph"
    elif mode == "full_highlight":
        title = "Full SCATS Routing Graph with Best Path Highlighted"
    elif mode == "top_k":
        title = "Top-k Routes Subgraph with Best Path Highlighted"
    elif mode == "best":
        title = "Best Path"
    else:
        title = "SCATS Routing Graph"

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return str(output_path)