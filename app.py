import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

airline_mapping = {
    'air asia': 0,
    'air india': 1,
    'go first': 2,
    'indigo': 3,
    'spicejet': 4,
    'vistara': 5
}

city_mapping = {
    'banglore': 0,
    'chennai': 1,
    'delhi': 2,
    'hyderabad': 3,
    'kolkata': 4,
    'mumbai': 5
}

class_mapping = {
    'business': 0,
    'economy': 1
}

time_category_mapping = {
    'afternoon': 0,
    'early morning': 1,
    'evening': 2,
    'late night': 3,
    'morning': 4,
    'night': 5
}

def classify_time_category(hour):
    if 0 <= hour < 4:
        return 'late night'
    elif 4 <= hour < 8:
        return 'early morning'
    elif 8 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 16:
        return 'afternoon'
    elif 16 <= hour < 20:
        return 'evening'
    else:
        return 'night'

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    airline = airline_mapping.get(request.form['airline'].lower(), 0)
    source_city = city_mapping.get(request.form['source_city'].lower(), 0)
    destination_city = city_mapping.get(request.form['destination_city'].lower(), 0)
    departure_hour = float(request.form['departure_time'])
    arrival_hour = float(request.form['arrival_time'])
    class1 = class_mapping.get(request.form['class'].lower(), 0)
    stops = float(request.form['stops'])
    duration = float(request.form['duration'])
    days_left = float(request.form['days_left'])

    departure_time_category = classify_time_category(departure_hour)
    arrival_time_category = classify_time_category(arrival_hour)

    departure_time_category_label = time_category_mapping[departure_time_category]
    arrival_time_category_label = time_category_mapping[arrival_time_category]

    final_features = np.array([[airline, source_city, departure_time_category_label, stops, arrival_time_category_label, destination_city, class1, duration, days_left]])
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='Ticket Price = {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
