import json
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from sklearn.datasets import load_wine
from streamlit.logger import get_logger

FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
LOW_CONFIDENCE = 0.6

LOGGER = get_logger(__name__)

@st.cache_data
def get_feature_ranges() -> dict:
    """Min, max and mean of each wine feature from the dataset, used for the sliders."""
    wine = load_wine()
    ranges = {}
    for i, name in enumerate(wine.feature_names):
        column = wine.data[:, i]
        # the API calls this feature od280_od315_of_diluted_wines (no slash)
        ranges[name.replace("/", "_")] = (float(column.min()), float(column.max()), float(column.mean()))
    return ranges

@st.cache_data(ttl=30)
def check_backend() -> str:
    """Returns 'online', 'error' or 'offline'. Cached for 30 seconds so the
    backend isn't called on every slider change."""
    try:
        response = requests.get(FASTAPI_BACKEND_ENDPOINT, timeout=5)
        return "online" if response.status_code == 200 else "error"
    except requests.RequestException as e:
        LOGGER.error(f"Backend error: {e}")
        return "offline"

def wine_sliders() -> dict:
    """One slider per feature, starting at the dataset average."""
    values = {}
    for name, (low, high, mean) in get_feature_ranges().items():
        # magnesium and proline are large whole numbers, the rest are small decimals
        step = 1.0 if high > 50 else 0.01
        start = round(mean) if step == 1.0 else round(mean, 2)
        label = name.replace("_", " ").capitalize()
        values[name] = st.slider(label, low, high, float(start), step)
    return values

def predict_wine(client_input: dict, source: str) -> None:
    """Send prediction request, display result and add it to the history."""
    result_container = st.empty()
    try:
        with st.spinner('Predicting...'):
            response = requests.post(
                f'{FASTAPI_BACKEND_ENDPOINT}/predict',
                json=client_input,
                timeout=10
            )

        if response.status_code == 200:
            content = response.json()
            confidence = content['confidence']
            with result_container.container():
                col1, col2 = st.columns(2)
                col1.metric("Predicted cultivar", content['class_name'])
                col2.metric("Confidence", f"{confidence:.0%}")
                st.progress(confidence)
                if confidence < LOW_CONFIDENCE:
                    st.warning("Low confidence: the model is unsure about this wine, treat the prediction with caution.")
            # keep every successful prediction for this browser session
            st.session_state["history"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "input": source,
                "cultivar": content['class_name'],
                "confidence": round(confidence, 3),
                **client_input,
            })
        elif response.status_code == 422:
            result_container.error("The backend rejected the input. All 13 features must be present and greater than 0.")
        else:
            st.toast(f':red[Status: {response.status_code}. Check backend]', icon="🔴")
    except Exception as e:
        st.toast(':red[Backend error. Refresh and retry]', icon="🔴")
        LOGGER.error(f"Prediction error: {e}")

def show_history() -> None:
    """Table of this session's predictions, with download and clear buttons."""
    history = st.session_state["history"]
    if not history:
        return
    st.subheader(f"Prediction history ({len(history)})")
    history_df = pd.DataFrame(history)
    st.dataframe(history_df, hide_index=True)
    col1, col2 = st.columns(2)
    col1.download_button("Download as CSV", history_df.to_csv(index=False),
                         file_name="prediction_history.csv", mime="text/csv")
    if col2.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()

def run():
    st.set_page_config(page_title="Wine Cultivar Prediction Demo", page_icon="🍷")

    # survives reruns, but starts empty again when the page is refreshed
    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.sidebar:
        # Backend status check
        backend_status = check_backend()
        if backend_status == "online":
            st.success("Backend online ✅")
        elif backend_status == "error":
            st.warning("Backend is up but returned an error 😭")
        else:
            st.error("Backend offline 😱")

        if st.button("Check again"):
            check_backend.clear()
            st.rerun()

        st.info("Configure parameters")

        input_mode = st.radio("Input method", ["Sliders", "Upload JSON"], horizontal=True)
        client_input = None

        if input_mode == "Sliders":
            client_input = wine_sliders()
        else:
            test_input_file = st.file_uploader('Upload test prediction file', type=['json'])
            if test_input_file:
                try:
                    test_input_data = json.load(test_input_file)
                    client_input = test_input_data['input_test']
                    st.write('Preview file')
                    st.json(test_input_data)
                except (ValueError, KeyError, TypeError):
                    st.error('File must be valid JSON with an "input_test" section')

        # nothing to send until there is input and the backend is reachable
        predict_button = st.button('Predict', disabled=client_input is None or backend_status != "online")

    st.write("# Wine Cultivar Prediction! 🍷")

    if predict_button:
        predict_wine(client_input, input_mode)

    show_history()

if __name__ == "__main__":
    run()