from routing.graph import load_graph
from routing.astar import find_route
from routing.graph_visualization import generate_route_graph_image


graph = load_graph()

result = find_route(
    graph=graph,
    origin="2000",
    destination="3002",
    departure_time="2006-10-01 08:00:00",
    model="lstm"
)

image_path = generate_route_graph_image(
    graph=graph,
    route_result=result,
    output_path="static/routing_graph.png"
)

print("Generated image:", image_path)