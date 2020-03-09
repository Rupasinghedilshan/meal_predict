import pickle
import os
import pandas as pd
import numpy as np
import csv
import json

from flask import Flask, jsonify, request
from flask import render_template
from waitress import serve

app = Flask(__name__)

reg_model = pickle.load(open('/home/sysadmin/Desktop/Angello/Ashen/meal_predict-master/num_orders.pkl', 'rb'))

@app.route('/predict',methods=['POST'])
def predict():

    req = request.get_json()

    oid = req['oid']
    week = req['week']
    totalprice = req['totalprice']
    category_Beverages = req['category_Beverages']
    category_Biriyani = req['category_Biriyani']
    category_Desert = req['category_Desert']
    category_Extras = req['category_Extras']
    category_Fish = req['category_Fish']
    category_OtherSnacks = req['category_OtherSnacks']
    category_Pasta = req['category_Pasta']
    category_Pizza = req['category_Pizza']
    category_RiceBowl = req['category_RiceBowl']
    category_Salad = req['category_Salad']
    category_Sandwich = req['category_Sandwich']
    category_Seafood = req['category_Seafood']
    category_Starters = req['category_Starters']
    cuisine_Continential = req['cuisine_Continential']
    cuisine_Indian = req['cuisine_Indian']
    cuisine_Italian = req['cuisine_Italian']
    cuisine_Thai = req['cuisine_Thai']

    week = int(week)
    totalprice = int(totalprice)

    num_orders_pred = reg_model.predict([[oid,week,totalprice,category_Beverages,category_Biriyani,category_Desert,category_Extras,category_Fish,category_OtherSnacks,category_Pasta,category_Pizza,category_RiceBowl,category_Salad,category_Sandwich,category_Seafood,category_Starters,cuisine_Continential,cuisine_Indian,cuisine_Italian,cuisine_Thai]])[0]

    return jsonify({'num_orders':num_orders_pred})


if __name__ == "__main__":
        #app.run(debug=True)
        serve(app, port = 8090)