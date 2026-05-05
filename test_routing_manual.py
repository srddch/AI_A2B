import random
from routing.graph import load_graph
from routing.astar import find_route

random.seed(42)  # 设置随机种子以获得可重复的结果
graph = load_graph()

test_cases = [
    ("2000", "3002"),
    ("2000", "4035"),
    ("3682", "3002"),
    ("4040", "3002"),
    ("9999", "3002"),
    ("3002", "2000")
]

for origin, destination in test_cases:
    print("=" * 60)
    print(f"Testing route: {origin} -> {destination}")

    result = find_route(
        graph=graph,
        origin=origin,
        destination=destination,
        departure_time="2006-10-01 08:00:00",
        model="lstm"
    )

    print("Best path:")
    print(result["best_path"])
    print("\nA* best path:")
    print(result.get("astar_best_path"))

    if result["paths"]:
        costs = [path["cost"] for path in result["paths"]]

        passed = (
            len(result["paths"]) <= 5
            and result["best_path"] == result["paths"][0]
            and result["best_path_index"] == 0
            and costs == sorted(costs)
            and result.get("astar_best_path") == result["best_path"]
        )
    else:
        passed = (
            result["best_path"] is None
            and result["best_path_index"] is None
        )

    

    print("\nAll paths:")
    for index, path in enumerate(result["paths"], start=1):
        print(f"{index}. {path['nodes']} | cost = {path['cost']:.2f} min")

    print("\nBest path index:")
    print(result["best_path_index"])

    print("\nTest result:")
    print("PASS" if passed else "FAIL")