from flask import Flask, request, jsonify, render_template

import pickle as pkl

app = Flask(__name__)
with open("model.pkl", "rb") as f:
    model = pkl.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            f1 = float(request.form["feature1"])
            f2 = float(request.form["feature2"])
            features = [[f1, f2]]
            prediction = model.predict(features)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["POST"])
def predict_web():
    f1 = float(request.form["feature1"])
    f2 = float(request.form["feature2"])
    features = [[f1, f2]]
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run()

