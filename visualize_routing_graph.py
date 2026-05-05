import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from routing.graph import load_graph
from routing.astar import find_route


def main():
    graph_data = load_graph()

    origin = "2000"
    destination = "3002"
    departure_time = "2006-10-01 08:00:00"
    model = "lstm"

    result = find_route(
        graph=graph_data,
        origin=origin,
        destination=destination,
        departure_time=departure_time,
        model=model
    )

    best_path = result["best_path"]["nodes"] if result["best_path"] else []

    G = nx.DiGraph()

    for node in graph_data["nodes"]:
        G.add_node(str(node))

    for edge in graph_data["edges"]:
        from_node = str(edge["from"])
        to_node = str(edge["to"])
        distance = edge["features"]["distance_km"]

        G.add_edge(from_node, to_node, distance=distance)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 9))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=900,
        node_color="lightgray"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=18,
        width=1.5
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold"
    )

    # Highlight best path
    if best_path:
        best_edges = list(zip(best_path[:-1], best_path[1:]))

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

    title = f"Routing Graph with Best Path: {origin} -> {destination}"
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    output_path = Path("routing_graph_visualization.png")
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved graph visualization to: {output_path}")


if __name__ == "__main__":
    main()