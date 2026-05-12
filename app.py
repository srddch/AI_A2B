from flask import Flask, render_template, request

from routing.astar import find_route
from routing.graph import load_graph
from routing.graph_visualization import generate_route_graph_image

import time

app = Flask(__name__)

graph = load_graph()


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    graph_image = None

    if request.method == "POST":

        origin = request.form.get("origin")
        destination = request.form.get("destination")

        # 日期和时间下拉框
        date = request.form.get("date")
        hour = request.form.get("hour")

        # 拼接成模型需要的格式，例如：2006-10-01 08:00:00
        departure_time = f"{date} {hour}:00:00"

        # 模型选择
        model = request.form.get("model")

        result = find_route(
            graph=graph,
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            model=model
        )

        graph_image = generate_route_graph_image(
            graph=graph,
            route_result=result,
            output_path="static/routing_graph.png"
        )

    return render_template(
        "index.html",
        result=result,
        graph_image=graph_image,
        time=time
    )


if __name__ == "__main__":
    app.run(debug=True)