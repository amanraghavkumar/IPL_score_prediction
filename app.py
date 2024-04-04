from flask import Flask, request,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

scaler=pickle.load(open("model/ipl_scaler.pkl", "rb"))
model = pickle.load(open("model/ipl_knr.pkl", "rb"))

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def Predict_datapoint():
    result=""

    if request.method=='POST':

        venue=int(request.form.get("venue"))
        bat_team = float(request.form.get('bat_team'))
        bowl_team = float(request.form.get('bowl_team'))
        runs = float(request.form.get('runs'))
        wickets = float(request.form.get('wickets'))
        overs = float(request.form.get('overs'))

        new_data=scaler.transform([[venue,bat_team,bowl_team,runs,wickets,overs]])
        result=model.predict(new_data)
            
        return render_template('result.html',result=result)


    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")