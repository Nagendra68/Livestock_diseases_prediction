from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import os
import pickle  # To load the trained model

app = Flask(__name__)

# Load the trained model
model_path = 'model.pkl'  # Path to your saved model
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None  # Handle the case where the model is not loaded
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Load the data
DATA_PATH = os.path.join('..','dataset', "animal_disease_dataset.csv")
data = pd.read_csv(DATA_PATH)

# Get animal name from the dataset to display on the form
animal_mapping = {index: label for index, label in enumerate(data['Animal'].unique())}

# Route to display the form
@app.route('/')
def index():
    return render_template('index.html', animal_mapping=animal_mapping)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'})

    try:
        # Get the data from the POST request.
        animal = int(request.form['animal'])
        age = float(request.form['age'])
        temperature = float(request.form['temperature'])

        symptoms = {
                'blisters_on_gums': int(request.form.get('blisters_on_gums', 0)),
                'blisters_on_hooves': int(request.form.get('blisters_on_hooves', 0)),
                'blisters_on_mouth': int(request.form.get('blisters_on_mouth', 0)),
                'blisters_on_tongue': int(request.form.get('blisters_on_tongue', 0)),
                'chest_discomfort': int(request.form.get('chest_discomfort', 0)),
                'chills': int(request.form.get('chills', 0)),
                'crackling_sound': int(request.form.get('crackling_sound', 0)),
                'depression': int(request.form.get('depression', 0)),
                'difficulty_walking': int(request.form.get('difficulty_walking', 0)),
                'fatigue': int(request.form.get('fatigue', 0)),
                'lameness': int(request.form.get('lameness', 0)),
                'loss_of_appetite': int(request.form.get('loss_of_appetite', 0)),
                'painless_lumps': int(request.form.get('painless_lumps', 0)),
                'shortness_of_breath': int(request.form.get('shortness_of_breath', 0)),
                'sores_on_gums': int(request.form.get('sores_on_gums', 0)),
                'sores_on_hooves': int(request.form.get('sores_on_hooves', 0)),
                'sores_on_mouth': int(request.form.get('sores_on_mouth', 0)),
                'sores_on_tongue': int(request.form.get('sores_on_tongue', 0)),
                'sweats': int(request.form.get('sweats', 0)),
                'swelling_in_abdomen': int(request.form.get('swelling_in_abdomen', 0)),
                'swelling_in_extremities': int(request.form.get('swelling_in_extremities', 0)),
                'swelling_in_limb': int(request.form.get('swelling_in_limb', 0)),
                'swelling_in_muscle': int(request.form.get('swelling_in_muscle', 0)),
                'swelling_in_neck': int(request.form.get('swelling_in_neck', 0))
                }

        # Prepare the feature list for prediction
        features = [
            animal, age, temperature, symptoms['blisters_on_gums'], symptoms['blisters_on_hooves'],
            symptoms['blisters_on_mouth'], symptoms['blisters_on_tongue'], symptoms['chest_discomfort'],
            symptoms['chills'], symptoms['crackling_sound'], symptoms['depression'],
            symptoms['difficulty_walking'], symptoms['fatigue'], symptoms['lameness'],
            symptoms['loss_of_appetite'], symptoms['painless_lumps'], symptoms['shortness_of_breath'],
            symptoms['sores_on_gums'], symptoms['sores_on_hooves'], symptoms['sores_on_mouth'],
            symptoms['sores_on_tongue'], symptoms['sweats'], symptoms['swelling_in_abdomen'],
            symptoms['swelling_in_extremities'], symptoms['swelling_in_limb'],
            symptoms['swelling_in_muscle'], symptoms['swelling_in_neck']
            ]
        
        # Convert the values to a numpy array
        feature_array = np.array([features])

        # Make prediction
        prediction = model.predict(feature_array)

        # Reverse map the prediction
        reverse_mapping = {label: index for index, label in enumerate(data['Disease'].unique())}
        predicted_disease = [key for key, value in reverse_mapping.items() if value == prediction[0]][0]

        return jsonify({'disease': str(predicted_disease)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
