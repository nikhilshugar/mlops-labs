import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
MIN_ROWS = 100
KMEANS_KWARGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")


def load_data(filename: str = "file.csv"):
    """
    Loads a CSV from the data folder and returns it as a base64 string (JSON-safe for XCom).
    """
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    print(f"Loaded {filename}: {len(df)} rows")
    return base64.b64encode(pickle.dumps(df)).decode("ascii")


def validate_data(data_b64: str):
    """
    Checks the loaded data before training.
    Returns a list of problems, empty if the data is fine.
    """
    df = pickle.loads(base64.b64decode(data_b64))
    problems = []

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        return problems

    usable = df.dropna()
    if len(usable) < MIN_ROWS:
        problems.append(f"only {len(usable)} usable rows, need at least {MIN_ROWS}")
    if (usable[FEATURES] < 0).any().any():
        problems.append("negative values in BALANCE, PURCHASES or CREDIT_LIMIT")

    if problems:
        print(f"Validation failed: {problems}")
    else:
        print(f"Validation passed: {len(usable)} usable rows")
    return problems


def stop_on_invalid_data(data_b64: str):
    """
    Runs only when validation fails. Fails the run with the list of problems.
    """
    problems = validate_data(data_b64)
    raise ValueError("Data validation failed: " + "; ".join(problems))


def data_preprocessing(data_b64: str):
    """
    Drops rows with missing values, keeps the 3 feature columns and scales them to 0-1.
    Saves the fitted scaler so the test data can be scaled the same way later.
    Returns the scaled data as a base64 string.
    """
    df = pickle.loads(base64.b64decode(data_b64))
    df = df.dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return base64.b64encode(pickle.dumps(scaled_data)).decode("ascii")


def find_optimal_k(data_b64: str, max_k: int = 49):
    """
    Fits K-Means for k = 1 to max_k and uses the elbow method to pick the best k.
    Returns k as a plain int so the next task can use it.
    """
    data = pickle.loads(base64.b64decode(data_b64))

    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, **KMEANS_KWARGS)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(range(1, max_k + 1), sse, curve="convex", direction="decreasing")
    if kl.elbow is None:
        raise ValueError("Elbow method could not find a good k")
    print(f"Optimal no. of clusters: {kl.elbow}")
    return int(kl.elbow)


def train_final_model(data_b64: str, k: int, filename: str):
    """
    Trains the final K-Means model with the k found by the elbow method and saves it.
    """
    data = pickle.loads(base64.b64decode(data_b64))

    kmeans = KMeans(n_clusters=k, **KMEANS_KWARGS)
    kmeans.fit(data)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(kmeans, f)
    print(f"Saved model with {k} clusters as {filename}")


def predict_test(filename: str):
    """
    Scales test.csv with the saved scaler and predicts a cluster for each row.
    """
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    predictions = [int(p) for p in model.predict(scaler.transform(test_df[FEATURES]))]
    print(f"Predicted clusters for test data: {predictions}")
    return predictions