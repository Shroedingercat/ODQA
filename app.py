import yaml

from Models.DPR import DPREncoder
from flask import Flask, render_template, request, url_for, redirect
app = Flask(__name__)


@app.route('/', methods=["POST", "GET"])
def main():
    if request.method == "POST":
        query = request.form["query"]
        return redirect(url_for("search", query=query))
    else:
        return render_template("index.html")


@app.route('/search/<query>', methods=["POST", "GET"])
def search(query):
    if request.method == "POST":
        query = request.form["query"]
        return redirect(url_for("search", query=query))
    else:
        answers = get_answers(query)
        return render_template("documents.html",
                               query=query, answers=answers)


def get_answers(query):
    if app.model is None:
        app.model = DPREncoder(question_model=app.config_yaml["retriever"]["model"],
                               config=app.config_yaml)
        app.model.create_faiss(app.config_yaml["retriever"]["index"],
                               n_jobs=app.config_yaml["retriever"]["n_jobs"])
    docs = app.model.retrieve(query, 10)

    answers = [["Answer: " + doc[0], doc[1], doc[2],
               f"https://en.wikipedia.org/wiki/{'_'.join(doc[1].split(' '))}"] for doc in docs]
    return answers


if __name__ == '__main__':
    app.model = None
    app.config_yaml = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

    app.run(host='0.0.0.0', port=5000, debug=True)