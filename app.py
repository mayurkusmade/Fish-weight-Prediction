from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model and label encoder
model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract user input
    species = request.form['species']
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])

    # Encode species
    species_encoded = label_encoder.transform([species])[0]

    # Prepare input data
    input_data = np.array([[species_encoded, length1, length2, length3, height, width]])

    # Predict weight
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Estimated Fish Weight: {prediction:.2f} grams')

if __name__ == "__main__":
    app.run(debug=True)
