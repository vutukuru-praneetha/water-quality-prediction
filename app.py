%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide"
)

preprocessor = joblib.load("water_preprocessor.pkl")
selector = joblib.load("water_feature_selector.pkl")
model = joblib.load("water_best_model.pkl")
model_info = joblib.load("water_model_info.pkl")

feature_names = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

st.title("💧 Explainable AI-Based Water Quality Prediction")
st.markdown(
    "Enter the water quality parameters below to predict whether the water is potable."
)

st.subheader("Water Quality Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1
    )

    hardness = st.number_input(
        "Hardness",
        min_value=0.0,
        value=200.0,
        step=1.0
    )

    solids = st.number_input(
        "Solids",
        min_value=0.0,
        value=12000.0,
        step=100.0
    )

with col2:
    chloramines = st.number_input(
        "Chloramines",
        min_value=0.0,
        value=7.0,
        step=0.1
    )

    sulfate = st.number_input(
        "Sulfate",
        min_value=0.0,
        value=300.0,
        step=1.0
    )

    conductivity = st.number_input(
        "Conductivity",
        min_value=0.0,
        value=400.0,
        step=1.0
    )

with col3:
    organic_carbon = st.number_input(
        "Organic Carbon",
        min_value=0.0,
        value=12.0,
        step=0.1
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes",
        min_value=0.0,
        value=60.0,
        step=0.1
    )

    turbidity = st.number_input(
        "Turbidity",
        min_value=0.0,
        value=4.0,
        step=0.1
    )

st.divider()

predict_button = st.button(
    "🔍 Predict Water Quality",
    use_container_width=True
)

if predict_button:

    new_water = pd.DataFrame({
        "ph": [ph],
        "Hardness": [hardness],
        "Solids": [solids],
        "Chloramines": [chloramines],
        "Sulfate": [sulfate],
        "Conductivity": [conductivity],
        "Organic_carbon": [organic_carbon],
        "Trihalomethanes": [trihalomethanes],
        "Turbidity": [turbidity]
    })

    new_water = new_water[feature_names]

    try:

        processed_data = preprocessor.transform(new_water)

        expected_features = model.n_features_in_

        if processed_data.shape[1] != expected_features:

            if (
                selector.get_support().sum() == expected_features
                and processed_data.shape[1] == len(feature_names)
            ):
                processed_data = selector.transform(processed_data)
            else:
                raise ValueError(
                    f"Feature mismatch: processed data has "
                    f"{processed_data.shape[1]} features, "
                    f"but model expects {expected_features}."
                )

        if processed_data.shape[1] != expected_features:
            raise ValueError(
                f"Final feature mismatch: "
                f"{processed_data.shape[1]} features received, "
                f"{expected_features} expected."
            )

        prediction = model.predict(processed_data)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_data)[0]
            not_potable_probability = probabilities[0] * 100
            potable_probability = probabilities[1] * 100
        else:
            probabilities = None
            not_potable_probability = None
            potable_probability = None

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("💧 WATER IS POTABLE")

            if potable_probability is not None:
                st.metric(
                    "Potable Probability",
                    f"{potable_probability:.2f}%"
                )

        else:

            st.error("⚠️ WATER IS NOT POTABLE")

            if not_potable_probability is not None:
                st.metric(
                    "Not Potable Probability",
                    f"{not_potable_probability:.2f}%"
                )

        if probabilities is not None:

            st.subheader("Prediction Probabilities")

            probability_data = pd.DataFrame({
                "Class": [
                    "Not Potable",
                    "Potable"
                ],
                "Probability (%)": [
                    not_potable_probability,
                    potable_probability
                ]
            })

            st.bar_chart(
                probability_data.set_index("Class")
            )

        st.subheader("Input Values")

        display_data = new_water.T.reset_index()
        display_data.columns = ["Parameter", "Value"]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Model Information")

        if isinstance(model_info, dict):

            if "best_model_name" in model_info:
                st.write(
                    "**Final Model:**",
                    model_info["best_model_name"]
                )

            if "feature_selection" in model_info:
                st.write(
                    "**Feature Selection:**",
                    model_info["feature_selection"]
                )

        st.info(
            "This prediction is generated using the trained machine learning "
            "model from the Water Quality Prediction project."
        )

    except Exception as e:

        st.error(
            f"Prediction error: {str(e)}"
        )
