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
MAX_BATCH = 1000

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

@app.route('/api/predict', methods=['POST'])
def predict_batch():
    """
    Predicts many flowers in one request.
    Body: {"flowers": [[sepal_length, sepal_width, petal_length, petal_width], ...]}
    """
    data = request.get_json(silent=True)
    if not data or "flowers" not in data:
        return jsonify({"error": 'send JSON like {"flowers": [[5.1, 3.5, 1.4, 0.2]]}'}), 400

    try:
        features = np.array(data["flowers"], dtype=float)
    except (ValueError, TypeError):
        return jsonify({"error": "all values must be numbers"}), 400

    if features.ndim != 2 or features.shape[1] != 4:
        return jsonify({"error": "each flower needs exactly 4 numbers"}), 400
    if len(features) > MAX_BATCH:
        return jsonify({"error": f"send at most {MAX_BATCH} flowers per request"}), 400

    # one model call for the whole batch instead of one request per flower
    probabilities = model.predict(scaler.transform(features), verbose=0)
    results = [{"predicted_class": class_labels[int(p.argmax())],
                "confidence": round(float(p.max()), 3)} for p in probabilities]
    return jsonify({"count": len(results), "predictions": results})

# only used when running "python main.py" directly, the container uses gunicorn
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)