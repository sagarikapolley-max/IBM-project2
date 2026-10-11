
"""
Streamlit frontend for the Water Quality Predictor (SDG 6).

Run with:
    python -m pip install -r requirements.txt
    python -m streamlit run app.py
"""

import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError as error:
    if error.name != "streamlit":
        raise
    raise SystemExit(
        "Streamlit is not installed in this Python environment.\n"
        "From the water_quality_app folder, run:\n"
        "  python -m pip install -r requirements.txt\n"
        "Then start the app with:\n"
        "  python -m streamlit run app.py"
    ) from error

import backend as be

st.set_page_config(page_title="Water Quality Predictor", page_icon="💧", layout="wide")


# ---------------------------------------------------------------
# Model loading (cached so it is not reloaded on every click)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading saved model...")
def load_saved_bundle():
    if be.MODEL_PATH.exists():
        try:
            return be.load_bundle()
        except Exception:
            return None
    return None


def get_bundle():
    """Use the saved model if there is one, otherwise train it from the CSV."""
    bundle = load_saved_bundle()
    if bundle is not None:
        return bundle

    if be.DEFAULT_CSV.is_file():
        source = be.DEFAULT_CSV
    else:
        st.info("`water_potability.csv` was not found next to app.py. Upload it to begin.")
        source = st.file_uploader("Upload water_potability.csv", type="csv")
        if source is None:
            st.stop()

    try:
        df = be.load_dataset(source)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    with st.spinner("Training the model (this takes a few seconds)..."):
        bundle = be.train(df)
        be.save_bundle(bundle)
    load_saved_bundle.clear()
    return bundle


bundle = get_bundle()

# ---------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------
st.title("💧 Water Quality Predictor")
st.caption("SDG 6: Clean Water and Sanitation · A machine learning learning project")

with st.sidebar:
    st.header("About")
    st.write(
        "Enter the measured properties of a water sample and the model "
        "estimates whether it is **potable** (safe to drink) or not."
    )
    st.metric("Model accuracy (test set)", f"{bundle.metrics['accuracy']:.1%}")
    st.caption(
        f"Random Forest trained on {bundle.metrics['train_size']} samples, "
        f"tested on {bundle.metrics['test_size']}."
    )
    st.divider()
    threshold = st.slider(
        "Decision threshold",
        0.20, 0.80, 0.50, 0.05,
        help="The sample is called potable when the potable probability is at "
             "least this value. Lower it to catch more potable samples; raise "
             "it to be stricter.",
    )
    if st.button("Retrain model"):
        be.MODEL_PATH.unlink(missing_ok=True)
        load_saved_bundle.clear()
        st.rerun()

tab_predict, tab_model, tab_about = st.tabs(["🔮 Predict", "📊 Model performance", "ℹ️ About"])

# ---------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------
with tab_predict:
    st.subheader("Enter water sample values")
    st.caption("Sliders start at the typical (median) value and stay within the range seen in the training data.")

    with st.form("sample_form"):
        values = {}
        columns = st.columns(3)
        for i, feature in enumerate(be.FEATURES):
            label, unit, help_text = be.FEATURE_INFO[feature]
            stats = bundle.feature_stats[feature]
            lo, hi = stats["min"], stats["max"]
            step = max((hi - lo) / 200, 0.01)
            with columns[i % 3]:
                values[feature] = st.slider(
                    f"{label} ({unit})" if unit else label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(stats["median"]),
                    step=float(step),
                    help=help_text,
                )
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        is_potable, probability = be.predict(bundle, values, threshold)

        st.divider()
        left, right = st.columns([2, 1])
        with left:
            if is_potable:
                st.success("### ✅ Likely potable\nThe model estimates this water may be safe to drink.")
            else:
                st.error("### ❌ Likely not potable\nThe model estimates this water may be unsafe to drink.")
        with right:
            st.metric("Potable probability", f"{probability:.0%}")
            st.progress(probability)

        st.warning(
            "This is a learning project. Real water safety must be confirmed "
            "by laboratory testing. Do not rely on this prediction to drink water."
        )

        with st.expander("Show the values used"):
            shown = pd.DataFrame(
                {
                    "Feature": [be.FEATURE_INFO[f][0] for f in be.FEATURES],
                    "Value": [values[f] for f in be.FEATURES],
                    "Unit": [be.FEATURE_INFO[f][1] for f in be.FEATURES],
                }
            )
            st.dataframe(shown, hide_index=True, width="stretch")

# ---------------------------------------------------------------
# Tab 2: Model performance
# ---------------------------------------------------------------
with tab_model:
    m = bundle.metrics
    potable = m["report"]["1"]
    not_potable = m["report"]["0"]

    st.subheader("How well does the model work?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.1%}")
    c2.metric("Potable precision", f"{potable['precision']:.0%}")
    c3.metric("Potable recall", f"{potable['recall']:.0%}")
    c4.metric("Potable F1-score", f"{potable['f1-score']:.0%}")
    st.caption(
        "Precision: of samples predicted potable, how many really were. "
        "Recall: of truly potable samples, how many the model found."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix** (test set)")
        cm = pd.DataFrame(
            m["confusion_matrix"],
            index=["Actual: Not potable", "Actual: Potable"],
            columns=["Predicted: Not potable", "Predicted: Potable"],
        )
        st.dataframe(cm, width="stretch")

        st.markdown("**Class balance** (all data)")
        counts = bundle.data_summary["class_counts"]
        st.bar_chart(
            pd.Series(
                {"Not potable (0)": counts.get(0, 0), "Potable (1)": counts.get(1, 0)}
            )
        )
    with right:
        st.markdown("**Feature importance** (which inputs matter most)")
        importance = pd.Series(m["importances"]).sort_values(ascending=False)
        importance.index = [be.FEATURE_INFO[f][0] for f in importance.index]
        st.bar_chart(importance)

        st.markdown("**Missing values in the dataset**")
        missing = pd.Series(bundle.data_summary["missing"])
        missing.index = [be.FEATURE_INFO[f][0] for f in missing.index]
        st.dataframe(missing.rename("Missing"), width="stretch")

# ---------------------------------------------------------------
# Tab 3: About
# ---------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Goal:** support **UN SDG 6 (Clean Water and Sanitation)** by showing how
machine learning can give a quick first estimate of drinking-water quality.

**How it works**
1. The backend (`backend.py`) loads `water_potability.csv`, fills missing values
   with the median, and trains a Random Forest classifier.
2. The model is saved to `model.joblib` so the app starts instantly next time.
3. The frontend (`app.py`) collects the nine water measurements and shows
   the predicted result and probability.

**Limitations**
- The dataset is small and hard to predict from, so accuracy is around 60-70%.
- The model finds potable water less reliably than non-potable water (see recall).
- It cannot detect bacteria, heavy metals or other contaminants not in the data.
- **Always confirm water safety with a certified laboratory test.**

**Ideas to improve it:** compare Logistic Regression and Gradient Boosting,
balance the classes with SMOTE, add data from your own region.
"""
    )