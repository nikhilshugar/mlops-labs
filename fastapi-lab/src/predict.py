import joblib

# load the model once when the server starts instead of on every request
model = joblib.load("../model/wine_model.pkl")

def predict_data(X):
    """
    Predict the class labels and class probabilities for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
        y_proba (numpy.ndarray): Predicted probability of each class.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, y_proba