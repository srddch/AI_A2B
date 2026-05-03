from flask import Flask, render_template, request
from routing.astar import find_route
from routing.graph import load_graph

app = Flask(__name__)
graph = load_graph()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        time = request.form.get("time")

        result = find_route(graph, origin, destination, time, "A*")

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)