from flask import Flask, render_template, request
from routing.top_k import find_top_k_paths

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        time = request.form.get("time")

        result = find_top_k_paths(origin, destination, time)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)