import json
import math
from pathlib import Path

import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def main():
    base_dir = Path(__file__).resolve().parent
    metadata_path = base_dir / "data" / "processed" / "site_metadata_merged.csv"
    output_path = base_dir / "data" / "processed" / "full_graph_data.json"

    df = pd.read_csv(metadata_path)

    # Keep only rows with valid coordinates.
    df = df.dropna(subset=["site_number", "latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

    sites = []

    for _, row in df.iterrows():
        sites.append(
            {
                "site_number": str(int(row["site_number"])),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
        )

    nodes = [site["site_number"] for site in sites]

    edges = []
    k_nearest = 3
    max_distance_km = 3.0

    for site in sites:
        distances = []

        for other in sites:
            if site["site_number"] == other["site_number"]:
                continue

            distance = haversine_km(
                site["latitude"],
                site["longitude"],
                other["latitude"],
                other["longitude"],
            )

            if distance <= max_distance_km:
                distances.append((other["site_number"], distance))

        distances.sort(key=lambda x: x[1])

        for to_site, distance in distances[:k_nearest]:
            edges.append(
                {
                    "from": site["site_number"],
                    "to": to_site,
                    "weight": round(distance, 3),
                    "features": {
                        "distance_km": round(distance, 3),
                        "generated_by": "nearest_neighbor",
                    },
                }
            )

    graph = {
        "nodes": nodes,
        "edges": edges,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    print(f"Generated graph: {output_path}")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")


if __name__ == "__main__":
    main()