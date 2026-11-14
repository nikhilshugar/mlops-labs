from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder='statics')

# settings come from environment variables so the Dockerfile and compose can set them
PORT = int(os.environ.get("PORT", 8000))
MODEL_PATH = os.environ.get("MODEL_PATH", "my_model.keras")
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.pkl")

# Load the TensorFlow model and the scaler it was trained with
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
class_labels = ['Setosa', 'Versicolor', 'Virginica']


@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            features = [float(data['sepal_length']), float(data['sepal_width']),
                        float(data['petal_length']), float(data['petal_width'])]

            # scale the inputs the same way as the training data
            input_data = scaler.transform(np.array([features]))
            prediction = model.predict(input_data, verbose=0)
            predicted_class = class_labels[int(np.argmax(prediction))]

            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=PORT)