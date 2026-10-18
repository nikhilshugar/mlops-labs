import pandas as pd
import requests
import streamlit as st

FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline",
]

def predict_batch(wines: pd.DataFrame) -> pd.DataFrame:
    """Send each row to the backend and return the table with the results added."""
    cultivars, confidences = [], []
    progress = st.progress(0.0, text="Predicting...")
    for i, row in enumerate(wines[FEATURES].to_dict(orient="records")):
        try:
            response = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}/predict", json=row, timeout=10)
        except requests.RequestException:
            st.error("Lost connection to the backend. Check that it is running and try again.")
            st.stop()
        if response.status_code == 200:
            content = response.json()
            cultivars.append(content["class_name"])
            confidences.append(content["confidence"])
        else:
            # e.g. 422 when a value is 0, negative or empty
            cultivars.append(f"error {response.status_code}")
            confidences.append(None)
        progress.progress((i + 1) / len(wines), text=f"Predicted {i + 1} of {len(wines)}")
    progress.empty()
    results = wines.copy()
    results.insert(0, "cultivar", cultivars)
    results.insert(1, "confidence", confidences)
    return results

def clear_results():
    # a new file means the old results no longer apply
    st.session_state.pop("batch_results", None)

st.set_page_config(page_title="Batch Prediction", page_icon="📋")
st.write("# Batch Prediction 📋")
st.write("Upload a CSV with one wine per row and a column for each of the 13 features. "
         "Every row is sent to the backend and the predictions are added to the table.")

uploaded_file = st.file_uploader("Upload wines CSV", type=["csv"], on_change=clear_results)

if uploaded_file:
    try:
        wines = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not read the file as a CSV.")
        st.stop()

    missing = [f for f in FEATURES if f not in wines.columns]
    if missing:
        st.error(f"The file is missing these columns: {', '.join(missing)}")
        st.stop()

    st.write(f"**{len(wines)} wines loaded**")
    st.dataframe(wines, hide_index=True)

    if st.button("Predict all"):
        st.session_state["batch_results"] = predict_batch(wines)

    results = st.session_state.get("batch_results")
    if results is not None:
        st.subheader("Results")
        st.dataframe(results, hide_index=True)
        st.bar_chart(results["cultivar"].value_counts())
        st.download_button("Download results as CSV", results.to_csv(index=False),
                           file_name="batch_predictions.csv", mime="text/csv")