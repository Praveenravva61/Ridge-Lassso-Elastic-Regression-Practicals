from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler= pickle.load(open("scaler.pkl", "rb"))
lin_reg= pickle.load(open("regressor.pkl", "rb"))

lasso_reg= pickle.load(open("Lasso.pkl", "rb"))

Ridge_reg= pickle.load(open("Ridge.pkl", "rb"))

applicaion= Flask(__name__)
app= applicaion

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))

        data = [[
            Temperature,
            RH,
            Ws,
            Rain,
            FFMC,
            DMC,
            ISI,
            Classes
        ]]

        
        scaled_data = scaler.transform(data)

        result = Ridge_reg.predict(scaled_data)

        return render_template(
            "home.html",
            prediction_text=round(result[0], 2)
        )

    return render_template("home.html")
            

if __name__ == "__main__":
    app.run(host= "0.0.0.0")
    
      

