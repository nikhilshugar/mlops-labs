from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Random Forest Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        RandomForestClassifier: The trained model.
    """
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=12)
    rf_classifier.fit(X_train, y_train)
    joblib.dump(rf_classifier, "../model/wine_model.pkl")
    return rf_classifier

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {accuracy:.3f}")