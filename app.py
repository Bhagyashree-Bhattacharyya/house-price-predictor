from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('models/LinearModel.pkl', 'rb'))
data = pd.read_csv('data/final_dataset.csv')
app = Flask(__name__)


features = sorted(['squareMeters', 'numberOfRooms', 'floors', 'numPrevOwners', 'ageOfHouse', 
            'cityPartRange', 'hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector', 
            'basement', 'attic', 'garage', 'hasStorageRoom', 'hasGuestRoom', 'cityCode'])
@app.route('/')
def index():
    return render_template('index.html', features=features)


@app.route('/predict', methods=['POST'])
def predict():
    squareMeters = request.form.get('squareMeters')
    numberOfRooms = request.form.get('numberOfRooms')
    floors = request.form.get('floors')
    numPrevOwners = request.form.get('numPrevOwners')
    ageOfHouse = request.form.get('ageOfHouse')
    cityPartRange = request.form.get('cityPartRange')
    hasYard = request.form.get('hasYard')
    hasPool = request.form.get('hasPool')
    isNewBuilt = request.form.get('isNewBuilt')
    hasStormProtector = request.form.get('hasStormProtector')
    basement = request.form.get('basement')
    attic = request.form.get('attic')
    garage = request.form.get('garage')
    hasStorageRoom = request.form.get('hasStorageRoom')
    hasGuestRoom = request.form.get('hasGuestRoom')
    cityCode = request.form.get('cityCode')

    input_data = pd.DataFrame([[squareMeters, numberOfRooms, floors, numPrevOwners, ageOfHouse,
                                 cityPartRange, hasYard, hasPool, isNewBuilt, hasStormProtector,
                                 basement, attic, garage, hasStorageRoom, hasGuestRoom, cityCode]],
                              columns=['squareMeters', 'numberOfRooms', 'floors', 'numPrevOwners', 'ageOfHouse',
                                       'cityPartRange', 'hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector',
                                       'basement', 'attic', 'garage', 'hasStorageRoom', 'hasGuestRoom', 'cityCode'])

    numeric_cols = ['squareMeters', 'numberOfRooms', 'floors', 'numPrevOwners', 'ageOfHouse', 'cityPartRange', 'cityCode']
    for col in numeric_cols:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

    binary_cols = ['hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector', 'basement', 'attic', 'garage', 'hasStorageRoom', 'hasGuestRoom']
    for col in binary_cols:
        input_data[col] = input_data[col].map({'yes': 1, 'no': 0})

    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])


    prediction = model.predict(input_data)[0]

    return str(prediction)




if __name__ == "__main__":
    app.run(debug=True, port=5000)