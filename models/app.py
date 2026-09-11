import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Adoption & CAFV Eligibility",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = BASE_DIR / "cafv_random_forest.pkl"
ENCODER_FILE = BASE_DIR / "categorical_encoders.pkl"
FEATURE_FILE = BASE_DIR / "feature_columns.pkl"
DATA_FILE = BASE_DIR / "ev_cleaned_dataset.csv"


# ============================================================
# CONSTANTS / MAPPINGS
# ============================================================

VEHICLE_VINTAGE_MAPPING = {
    "Mid-Age": 0,
    "New": 1,
    "Older": 2,
    "Recent": 3
}


CAFV_STATUS_MAPPING = {
    0: "Eligible",
    1: "Not Eligible",
    2: "Unknown"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE.name}"
        )

    # --------------------------------------------------------
    # Try joblib first
    # --------------------------------------------------------

    try:

        model = joblib.load(MODEL_FILE)

        return model

    except Exception as joblib_error:

        # ----------------------------------------------------
        # Try standard pickle
        # ----------------------------------------------------

        try:

            with open(MODEL_FILE, "rb") as file:
                model = pickle.load(file)

            return model

        except Exception as pickle_error:

            raise RuntimeError(
                "Unable to load the Random Forest model.\n\n"
                f"Joblib error: {joblib_error}\n\n"
                f"Pickle error: {pickle_error}\n\n"
                "The model file may be corrupted or may have been "
                "created using an incompatible serialization method."
            )


# ============================================================
# LOAD ENCODERS
# ============================================================

@st.cache_resource
def load_encoders():

    if not ENCODER_FILE.exists():
        raise FileNotFoundError(
            f"Encoder file not found: {ENCODER_FILE.name}"
        )

    # Try joblib first
    try:

        encoders = joblib.load(ENCODER_FILE)

        return encoders

    except Exception:

        # Try pickle
        with open(ENCODER_FILE, "rb") as file:

            encoders = pickle.load(file)

        return encoders


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

@st.cache_resource
def load_feature_columns():

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_FILE.name}"
        )

    # Try joblib first
    try:

        feature_columns = joblib.load(FEATURE_FILE)

        return feature_columns

    except Exception:

        # Try pickle
        with open(FEATURE_FILE, "rb") as file:

            feature_columns = pickle.load(file)

        return feature_columns


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE.name}"
        )

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    return df


# ============================================================
# LOAD ALL ARTIFACTS
# ============================================================

try:

    model = load_model()

    encoders = load_encoders()

    feature_columns = load_feature_columns()

    df = load_data()


except FileNotFoundError as e:

    st.error(
        "❌ Required file is missing."
    )

    st.error(
        str(e)
    )

    st.info(
        """
        Make sure these files are inside the same folder as app.py:

        - cafv_random_forest.pkl
        - categorical_encoders.pkl
        - feature_columns.pkl
        - ev_cleaned_dataset.csv
        """
    )

    st.stop()


except Exception as e:

    st.error(
        "❌ Unable to load the machine-learning files."
    )

    st.exception(e)

    st.warning(
        """
        The Random Forest model may have been saved using a different
        serialization method or the .pkl file may be corrupted.

        If this error continues, regenerate the model file using
        joblib.dump() or pickle.dump().
        """
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ EV Analytics")

st.sidebar.markdown(
    """
    **Electric Vehicle Adoption & CAFV Eligibility**

    Interactive analytics dashboard and
    machine-learning prediction system.
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Eligibility Prediction"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚡ Electric Vehicle Adoption & CAFV Eligibility"
)

st.markdown(
    "Explore EV adoption patterns and predict Clean Alternative Fuel "
    "Vehicle (CAFV) eligibility using a Random Forest model."
)

st.divider()


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.header("📊 EV Adoption Overview")


    # ========================================================
    # KPI CALCULATIONS
    # ========================================================

    total_evs = len(df)

    manufacturers = df["Make"].nunique()

    models = df["Model"].nunique()

    counties = df["County"].nunique()

    cities = df["City"].nunique()


    # ========================================================
    # KPI CARDS
    # ========================================================

    col1, col2, col3, col4, col5 = st.columns(5)


    col1.metric(
        "Total EV Records",
        f"{total_evs:,}"
    )


    col2.metric(
        "Manufacturers",
        f"{manufacturers:,}"
    )


    col3.metric(
        "EV Models",
        f"{models:,}"
    )


    col4.metric(
        "Counties",
        f"{counties:,}"
    )


    col5.metric(
        "Cities",
        f"{cities:,}"
    )


    st.divider()


    # ========================================================
    # EV TYPE + MANUFACTURERS
    # ========================================================

    col1, col2 = st.columns(2)


    # ========================================================
    # EV TYPE DISTRIBUTION
    # ========================================================

    with col1:

        st.subheader("EV Type Distribution")

        ev_type_counts = (
            df["Electric_Vehicle_Type"]
            .value_counts()
            .reset_index()
        )

        ev_type_counts.columns = [
            "Electric_Vehicle_Type",
            "Count"
        ]

        fig_ev_type = px.pie(
            ev_type_counts,
            names="Electric_Vehicle_Type",
            values="Count",
            hole=0.45,
            title="Battery EV vs Plug-in Hybrid EV"
        )

        fig_ev_type.update_layout(
            height=400
        )

        st.plotly_chart(
            fig_ev_type,
            use_container_width=True
        )


    # ========================================================
    # TOP MANUFACTURERS
    # ========================================================

    with col2:

        st.subheader("Top 10 EV Manufacturers")

        manufacturer_counts = (
            df["Make"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        manufacturer_counts.columns = [
            "Make",
            "Count"
        ]

        fig_manufacturers = px.bar(
            manufacturer_counts,
            x="Count",
            y="Make",
            orientation="h",
            title="Top 10 Manufacturers by EV Registrations"
        )

        fig_manufacturers.update_layout(
            height=400,
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig_manufacturers,
            use_container_width=True
        )


    st.divider()


    # ========================================================
    # CAFV ELIGIBILITY DISTRIBUTION
    # ========================================================

    st.subheader(
        "CAFV Eligibility Distribution"
    )

    cafv_counts = (
        df[
            "Clean_Alternative_Fuel_Vehicle_CAFV_Eligibility"
        ]
        .value_counts()
        .reset_index()
    )

    cafv_counts.columns = [
        "CAFV_Status",
        "Count"
    ]

    fig_cafv = px.bar(
        cafv_counts,
        x="CAFV_Status",
        y="Count",
        title="CAFV Eligibility Status",
        text="Count"
    )

    fig_cafv.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_cafv,
        use_container_width=True
    )


    # ========================================================
    # DATA SUMMARY
    # ========================================================

    st.subheader(
        "Dataset Summary"
    )

    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.metric(
            "Battery EVs",
            f"{(df['Is_BEV'] == 1).sum():,}"
        )

        st.metric(
            "Average Electric Range",
            f"{df['Electric_Range'].mean():.1f}"
        )


    with summary_col2:

        st.metric(
            "Premium Brand Vehicles",
            f"{(df['Premium_Brand'] == 1).sum():,}"
        )

        st.metric(
            "Average Vehicle Age",
            f"{df['Vehicle_Age'].mean():.1f}"
        )


# ============================================================
# ELIGIBILITY PREDICTION PAGE
# ============================================================

elif page == "Eligibility Prediction":

    st.header(
        "🔮 CAFV Eligibility Prediction"
    )

    st.markdown(
        """
        Enter vehicle details below to predict whether the vehicle
        is **Eligible**, **Not Eligible**, or **Unknown** for CAFV status.
        """
    )


    # ========================================================
    # INPUT SECTION
    # ========================================================

    st.subheader(
        "Vehicle Information"
    )


    col1, col2, col3 = st.columns(3)


    # ========================================================
    # COUNTY
    # ========================================================

    with col1:

        county_options = sorted(
            df["County"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_county = st.selectbox(
            "County",
            county_options
        )


    # ========================================================
    # CITY
    # ========================================================

    with col2:

        city_options = sorted(
            df["City"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_city = st.selectbox(
            "City",
            city_options
        )


    # ========================================================
    # MAKE
    # ========================================================

    with col3:

        make_options = sorted(
            df["Make"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_make = st.selectbox(
            "Manufacturer",
            make_options
        )


    # ========================================================
    # MODEL
    # ========================================================

    filtered_make = df[
        df["Make"].astype(str) == selected_make
    ]

    model_options = sorted(
        filtered_make["Model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_model = st.selectbox(
        "Vehicle Model",
        model_options
    )


    # ========================================================
    # EV TYPE
    # ========================================================

    filtered_model = filtered_make[
        filtered_make["Model"].astype(str) == selected_model
    ]

    ev_type_options = sorted(
        filtered_model["Electric_Vehicle_Type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_ev_type = st.selectbox(
        "Electric Vehicle Type",
        ev_type_options
    )


    # ========================================================
    # NUMERIC INPUTS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        electric_range = st.number_input(
            "Electric Range",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=1.0
        )


    with col2:

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=47.5,
            step=0.0001,
            format="%.4f"
        )


    with col3:

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-122.3,
            step=0.0001,
            format="%.4f"
        )


    # ========================================================
    # VEHICLE FEATURES
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        vehicle_age = st.number_input(
            "Vehicle Age",
            min_value=0,
            max_value=50,
            value=2,
            step=1
        )


    with col2:

        selected_vintage = st.selectbox(
            "Vehicle Vintage",
            list(
                VEHICLE_VINTAGE_MAPPING.keys()
            )
        )

        vehicle_vintage = (
            VEHICLE_VINTAGE_MAPPING[
                selected_vintage
            ]
        )


    with col3:

        premium_brand = st.selectbox(
            "Premium Brand",
            [
                "No",
                "Yes"
            ]
        )

        premium_brand_value = (
            1
            if premium_brand == "Yes"
            else 0
        )


    # ========================================================
    # BEV
    # ========================================================

    is_bev = st.selectbox(
        "Battery Electric Vehicle (BEV)",
        [
            "Yes",
            "No"
        ]
    )

    is_bev_value = (
        1
        if is_bev == "Yes"
        else 0
    )


    st.divider()


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "🔍 Model Information"
    ):

        try:

            st.write(
                "Model type:",
                type(model).__name__
            )

            if hasattr(model, "classes_"):

                st.write(
                    "Model classes:",
                    model.classes_
                )

            st.write(
                "Number of features:",
                len(feature_columns)
            )

            st.write(
                "Features:",
                feature_columns
            )

        except Exception:

            pass


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "🔮 Predict CAFV Eligibility",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            # =================================================
            # ENCODE CATEGORICAL VARIABLES
            # =================================================

            encoded_values = {}


            categorical_inputs = {

                "County":
                    selected_county,

                "City":
                    selected_city,

                "Make":
                    selected_make,

                "Model":
                    selected_model,

                "Electric_Vehicle_Type":
                    selected_ev_type

            }


            for column, value in categorical_inputs.items():

                if column not in encoders:

                    raise KeyError(
                        f"Encoder for '{column}' "
                        "was not found."
                    )

                encoder = encoders[column]


                try:

                    encoded_values[column] = (
                        encoder.transform(
                            [value]
                        )[0]
                    )

                except Exception:

                    st.error(
                        f"Unable to encode {column}: {value}"
                    )

                    st.info(
                        "This value may not have been present "
                        "when the encoder was trained."
                    )

                    st.stop()


            # =================================================
            # CREATE MODEL INPUT
            # =================================================

            input_data = {

                "County":
                    encoded_values["County"],

                "City":
                    encoded_values["City"],

                "Make":
                    encoded_values["Make"],

                "Model":
                    encoded_values["Model"],

                "Electric_Vehicle_Type":
                    encoded_values[
                        "Electric_Vehicle_Type"
                    ],

                "Electric_Range":
                    electric_range,

                "Longitude":
                    longitude,

                "Latitude":
                    latitude,

                "Vehicle_Age":
                    vehicle_age,

                "Vehicle_Vintage":
                    vehicle_vintage,

                "Premium_Brand":
                    premium_brand_value,

                "Is_BEV":
                    is_bev_value

            }


            input_df = pd.DataFrame(
                [input_data]
            )


            # =================================================
            # CHECK FEATURE COLUMNS
            # =================================================

            if not feature_columns:

                raise ValueError(
                    "feature_columns.pkl is empty."
                )


            missing_features = [
                col
                for col in feature_columns
                if col not in input_df.columns
            ]


            if missing_features:

                raise ValueError(
                    "The following model features are missing: "
                    + ", ".join(
                        missing_features
                    )
                )


            # =================================================
            # FORCE EXACT FEATURE ORDER
            # =================================================

            input_df = input_df[
                feature_columns
            ]


            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(
                input_df
            )[0]


            # =================================================
            # PREDICTION PROBABILITY
            # =================================================

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(
                        input_df
                    )[0]
                )

            else:

                probabilities = None


            # =================================================
            # GET PREDICTED STATUS
            # =================================================

            try:

                predicted_class = int(
                    prediction
                )

            except Exception:

                predicted_class = prediction


            predicted_status = (
                CAFV_STATUS_MAPPING.get(
                    predicted_class,
                    str(predicted_class)
                )
            )


            # =================================================
            # DISPLAY RESULT
            # =================================================

            st.divider()

            st.subheader(
                "Prediction Result"
            )


            if predicted_status == "Eligible":

                st.success(
                    "✅ Vehicle is predicted to be CAFV Eligible"
                )


            elif predicted_status == "Not Eligible":

                st.error(
                    "❌ Vehicle is predicted to be CAFV Not Eligible"
                )


            else:

                st.warning(
                    f"⚠️ Prediction: {predicted_status}"
                )


            # =================================================
            # RESULT METRICS
            # =================================================

            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Predicted Status",
                    predicted_status
                )


            with col2:

                if probabilities is not None:

                    # Find probability corresponding
                    # to the predicted class

                    confidence = None


                    if hasattr(
                        model,
                        "classes_"
                    ):

                        classes = list(
                            model.classes_
                        )

                        try:

                            class_index = (
                                classes.index(
                                    prediction
                                )
                            )

                            confidence = (
                                probabilities[
                                    class_index
                                ] * 100
                            )

                        except ValueError:

                            confidence = (
                                max(
                                    probabilities
                                ) * 100
                            )

                    else:

                        confidence = (
                            max(
                                probabilities
                            ) * 100
                        )


                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                else:

                    st.metric(
                        "Model Confidence",
                        "N/A"
                    )


            # =================================================
            # PROBABILITY CHART
            # =================================================

            if probabilities is not None:

                if hasattr(
                    model,
                    "classes_"
                ):

                    classes = list(
                        model.classes_
                    )

                    probability_rows = []


                    for i, class_value in enumerate(
                        classes
                    ):

                        try:

                            class_number = int(
                                class_value
                            )

                            status = (
                                CAFV_STATUS_MAPPING.get(
                                    class_number,
                                    str(class_value)
                                )
                            )

                        except Exception:

                            status = str(
                                class_value
                            )


                        probability_rows.append({

                            "Status":
                                status,

                            "Probability":
                                probabilities[i],

                            "Probability (%)":
                                probabilities[i] * 100

                        })


                    probability_df = pd.DataFrame(
                        probability_rows
                    )


                    fig_probability = px.bar(
                        probability_df,
                        x="Status",
                        y="Probability (%)",
                        title="Prediction Probability",
                        text="Probability (%)",
                        color="Status"
                    )


                    fig_probability.update_traces(
                        texttemplate="%{text:.2f}%",
                        textposition="outside"
                    )


                    fig_probability.update_layout(
                        yaxis_range=[
                            0,
                            100
                        ],
                        showlegend=False
                    )


                    st.plotly_chart(
                        fig_probability,
                        use_container_width=True
                    )


            # =================================================
            # MODEL INPUT PREVIEW
            # =================================================

            with st.expander(
                "👁️ View Model Input"
            ):

                st.dataframe(
                    input_df,
                    use_container_width=True
                )


            # =================================================
            # RAW PREDICTION DETAILS
            # =================================================

            with st.expander(
                "🔧 Technical Prediction Details"
            ):

                st.write(
                    "Raw prediction:",
                    prediction
                )

                if hasattr(
                    model,
                    "classes_"
                ):

                    st.write(
                        "Model classes:",
                        model.classes_
                    )

                if probabilities is not None:

                    st.write(
                        "Raw probabilities:",
                        probabilities
                    )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EV Adoption Analytics | "
    "Random Forest CAFV Eligibility Model"
)