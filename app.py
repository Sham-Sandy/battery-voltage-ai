from flask import Flask, render_template, request, jsonify
import os
from model_api import predict_voltage

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    formula = request.form["formula"]
    file = request.files["cif"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    voltage = predict_voltage(path, formula)

    return jsonify({"voltage": round(voltage,3)})

if __name__ == "__main__":
    app.run(debug=True)