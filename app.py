"""Diamond Price Predictor: Streamlit web app for the model trained in diamond_price_kfold_10fold.ipynb."""
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
MODEL_FILE = HERE / "diamond_price_model.pkl"
INFO_FILE = HERE / "model_info.json"

# In the dataset, the average diameter (x + y) / 2 is very close to 6.42 mm x carat^(1/3)
DIAMETER_PER_CUBE_ROOT_CARAT = 6.42

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_info():
    return json.loads(INFO_FILE.read_text())


model = load_model()
info = load_info()
ranges = info["numeric_ranges"]
categories = info["categories"]

# ---------------------------------------------------------------- header
st.title("💎 Diamond Price Predictor")
st.write(
    f"Estimate the price of a diamond from its 4 Cs and proportions. "
    f"The model is a **{info['algorithm']}** pipeline trained on {info['dataset']}, "
    f"selected by comparing 5 algorithms with {info['cross_validation']} cross validation."
)

# ---------------------------------------------------------------- inputs
with st.form("diamond"):
    st.subheader("Diamond attributes")

    carat = st.slider("Carat (weight)", min_value=0.20, max_value=5.01, value=1.00, step=0.01)

    col1, col2, col3 = st.columns(3)
    cut = col1.selectbox("Cut", categories["cut"], index=categories["cut"].index("Ideal"),
                         help="Fair (worst) → Ideal (best)")
    color = col2.selectbox("Colour", categories["color"], index=categories["color"].index("G"),
                           help="J (worst) → D (best, colourless)")
    clarity = col3.selectbox("Clarity", categories["clarity"], index=categories["clarity"].index("VS2"),
                             help="I1 (worst) → IF (internally flawless)")

    col4, col5 = st.columns(2)
    depth = col4.number_input("Depth (%)", min_value=float(ranges["depth"]["min"]),
                              max_value=float(ranges["depth"]["max"]), value=float(ranges["depth"]["median"]),
                              step=0.1, help="Total depth percentage = 2·z / (x + y)")
    table = col5.number_input("Table (%)", min_value=float(ranges["table"]["min"]),
                              max_value=float(ranges["table"]["max"]), value=float(ranges["table"]["median"]),
                              step=0.5, help="Width of the top facet relative to the widest point")

    exact_size = st.checkbox("I know the exact measurements (x, y, z in mm)",
                             help="Otherwise they are estimated from the carat and depth.")
    estimated_diameter = DIAMETER_PER_CUBE_ROOT_CARAT * carat ** (1 / 3)
    if exact_size:
        col6, col7, col8 = st.columns(3)
        x = col6.number_input("Length x (mm)", 0.1, float(ranges["x"]["max"]), round(estimated_diameter, 2), 0.01)
        y = col7.number_input("Width y (mm)", 0.1, float(ranges["y"]["max"]), round(estimated_diameter, 2), 0.01)
        z = col8.number_input("Depth z (mm)", 0.1, float(ranges["z"]["max"]),
                              round(estimated_diameter * depth / 100, 2), 0.01)
    else:
        x = y = estimated_diameter
        z = estimated_diameter * depth / 100

    submitted = st.form_submit_button("Estimate price", type="primary", width="stretch")

# ---------------------------------------------------------------- prediction
if submitted:
    diamond = {"carat": carat, "depth": depth, "table": table, "x": x, "y": y, "z": z,
               "cut": cut, "color": color, "clarity": clarity}
    features = pd.DataFrame([diamond])[info["numeric_features"] + info["categorical_features"]]
    price = max(float(model.predict(features)[0]), 0.0)

    st.metric("Estimated price", f"${price:,.0f}")
    st.caption(
        f"Typical error from 10-fold cross validation: about ±\\${info['cv_mae']:,.0f} (MAE), "
        f"RMSE \\${info['cv_rmse']:,.0f}. Dataset prices range from \\$326 to \\$18,823."
    )
    if not exact_size:
        st.caption(f"Measurements used (estimated from carat and depth): "
                   f"x = y = {x:.2f} mm, z = {z:.2f} mm.")
    if carat > 3:
        st.warning("Fewer than 0.1% of the training diamonds weigh more than 3 carats, "
                   "so this estimate is less reliable.")

# ---------------------------------------------------------------- model card
with st.expander("How the model was chosen"):
    comparison = pd.DataFrame(info["comparison"]).set_index("Rank")
    st.write(
        f"Five scikit-learn regression algorithms were evaluated on the same {info['cross_validation']} "
        f"folds with the same pre-processing and the same metrics."
    )
    st.write(info["selection_note"])
    st.dataframe(
        comparison.style.format({"Mean RMSE ($)": "${:,.0f}", "Std RMSE ($)": "${:,.0f}",
                                 "Mean MAE ($)": "${:,.0f}", "Mean R2": "{:.4f}", "Time (s)": "{:.1f}"}),
        width="stretch",
    )
    st.caption(f"Cross-validated R² of the deployed model: {info['cv_r2']:.4f}. "
               f"Trained with scikit-learn {info['scikit_learn_version']}.")
