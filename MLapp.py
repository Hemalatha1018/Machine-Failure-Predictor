import numpy as np
from flask import Flask, request, jsonify, render_template
from collections.abc import Mapping
import joblib  # Use joblib instead of pickle

app = Flask(__name__)
model = joblib.load('model (1).pkl')  # Load the model using joblib

@app.route('/')
def home():
    return render_template('pmc.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = [np.array(list(request.form.values()))]
    prediction = model.predict(data)
    output = prediction[0]
    return render_template('pmc.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
