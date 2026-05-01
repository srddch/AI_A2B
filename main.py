from ui.cli import get_user_input
from routing.top_k import find_top_k_paths

def main():
    origin, destination, time = get_user_input()

    routes = find_top_k_paths(origin, destination, time)

    print("\n=== Routes ===")
    for i, r in enumerate(routes):
        print(f"{i+1}. Path: {r['path']}, Time: {r['time']:.2f}s")

if __name__ == "__main__":
    main()
