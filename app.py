from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the trained model
with open("delivery_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [
        data["Agent_Age"],
        data["Agent_Rating"],
        data["Store_Latitude"],
        data["Store_Longitude"],
        data["Drop_Latitude"],
        data["Drop_Longitude"],
        data["Weather"],
        data["Traffic"],
        data["Vehicle"],
        data["Area"],
        data["Category"]
    ]

    prediction = model.predict([np.array(features)])[0]

    return jsonify({"predicted_delivery_time": prediction})

if __name__ == '__main__':
    app.run(debug=True)
