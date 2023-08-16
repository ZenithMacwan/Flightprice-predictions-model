import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html',prediction_text="")

@app.route('/predict',methods=['POST'])
def predict():
    airline = 0  # Default value
    if request.form['airline'] == 'Air Asia':
        airline = 0
    elif request.form['airline'] == 'Air India':
        airline = 1        
    elif request.form['airline'] == 'Go First':
        airline = 2
    elif request.form['airline'] == 'Indigo':
        airline = 3
    elif request.form['airline'] == 'SpiceJet':
        airline = 4
    elif request.form['airline'] == 'Vistara':
        airline = 5
    source_city = 0
    if request.form['source_city'] == 'Air Asia':
        airline = 0
    elif request.form['source_city'] == 'Air India':
        airline = 1        
    elif request.form['source_city'] == 'Go First':
        airline = 2
    elif request.form['source_city'] == 'Indigo':
        airline = 3
    elif request.form['source_city'] == 'SpiceJet':
        airline = 4
    elif request.form['source_city'] == 'Vistara':
        airline = 5

    source_city = float(request.form['source_city'])
    departure_time = float(request.form['departure_time'])
    stops = float(request.form['stops'])
    arrival_time = float(request.form['arrival_time'])
    destination_city = float(request.form['destination_city'])
    class1 = float(request.form['class'])
    duration = float(request.form['duration'])
    days_left = float(request.form['days_left'])
    final_features = [np.array([airline,source_city,departure_time,stops,arrival_time,destination_city,class1,duration,days_left])]
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='Car Price = {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)