'''
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from def1 import *

app = Flask(__name__)
pipeline = pickle.load(open('trained_model.pkl', 'rb'))

# Define the route for the prediction form
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    myDict={}
    # Retrieve the user's input from the form
    myDict['City'] = request.form.get('City')
    myDict['type'] = request.form.get('type')
    myDict['room_number'] = float(request.form.get('room_number'))
    myDict['area'] = float(request.form.get('Area'))
    myDict['street'] = request.form.get('Street')
    myDict['number_in_street'] = float(request.form.get('number_in_street'))
    myDict['city_area'] = request.form.get('city_area')
    myDict['num_of_images'] = float(request.form.get('num_of_images'))
    myDict['floor_out_of'] = request.form.get('floor_out_of')
    myDict['has_elevator'] = float(request.form.get('hasElevator'))
    myDict['has_parking'] = float(request.form.get('hasParking'))
    myDict['has_bars'] = float(request.form.get('hasBars'))
    myDict['has_storage'] = float(request.form.get('hasStorage'))
    myDict['condition'] = request.form.get('condition')
    myDict['has_air_condition'] = float(request.form.get('hasAirCondition'))
    myDict['has_balcony'] = float(request.form.get('hasBalcony'))
    myDict['has_mamad'] = float(request.form.get('hasMamad'))
    myDict['handicap_friendly'] = float(request.form.get('handicapFriendly'))
    myDict['entrance_date'] = request.form.get('entranceDate')
    myDict['furniture'] = request.form.get('furniture')
    myDict['published_days'] = request.form.get('publishedDays')
    myDict['description'] = request.form.get('description')
   

    df = pd.DataFrame([myDict])

    prediction = pipeline.predict(df)

    # Return the predicted price to the user
    return render_template('index.html', prediction_text='The home should be $ {}'.format(prediction))

  
if __name__ == "__main__": #when im running this code from somewhere else it won't run this
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)
'''
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
from def1 import *

app = Flask(__name__)
pipeline = pickle.load(open('trained_model.pkl', 'rb'))

# Define the route for the prediction form
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    myDict = {
        'City': [request.form['City']],
        'type': [request.form['type']],
        'room_number': [float(request.form['room_number'])],
        'Area': [float(request.form['Area'])],
        'Street': [request.form['Street']],
        'number_in_street': [float(request.form['number_in_street'])],
        'city_area': [request.form['city_area']],
        'num_of_images': [float(request.form['num_of_images'])],
        'floor': [request.form['floor']],
        'total_floors': [request.form['total_floors']],
        'hasElevator': [float(request.form['hasElevator'])],
        'hasParking': [float(request.form['hasParking'])],
        'hasBars': [float(request.form['hasBars'])],
        'hasStorage': [float(request.form['hasStorage'])],
        'condition': [request.form['condition']],
        'hasAirCondition': [float(request.form['hasAirCondition'])],
        'hasBalcony': [float(request.form['hasBalcony'])],
        'hasMamad': [float(request.form['hasMamad'])],
        'handicapFriendly': [float(request.form['handicapFriendly'])],
        'entrance_date': [request.form['entrance_date']],
        'furniture': [request.form['furniture']],
        'publishedDays': [float(request.form['publishedDays'])],
        'description': [request.form['description']]
    }

    prediction = pd.DataFrame(myDict)

    print("Before reshaping:\n", prediction)

    
    # Perform the prediction
    predicted_price = pipeline.predict(prediction)

    # Return the predicted price to the user
    return render_template('index.html', prediction_text='The home should be $ {}'.format(predicted_price))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

