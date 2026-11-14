
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_models():

    rf_model = joblib.load(
        "models/random_forest_model.pkl"
    )

    dnn_model = tf.keras.models.load_model(
        "models/student_performance_model.keras"
    )

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    target_mapping = joblib.load(
        "models/target_mapping.pkl"
    )

    return rf_model, dnn_model, preprocessor, target_mapping


rf_model, dnn_model, preprocessor, target_mapping = load_models()

# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🎓 Student Academic Performance Predictor")

st.markdown(
    """
    Predict student academic performance using
    **Machine Learning and Deep Learning**.

    Enter the student's information below to predict
    their academic performance category using two models.
    """
)

st.divider()

st.success(
    "Deep Learning and Machine Learning models loaded successfully!"
)

# ==========================================
# PHASE 16: STUDENT INPUT FORM
# ==========================================

st.header("📝 Enter Student Information")

st.write(
    "Fill in the details below to predict academic performance."
)

with st.form("student_form"):

    # ======================================
    # PERSONAL INFORMATION
    # ======================================

    st.subheader("1. Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        school = st.selectbox(
            "School",
            ["GP", "MS"]
        )

        sex = st.selectbox(
            "Gender",
            ["F", "M"]
        )

        age = st.number_input(
            "Age",
            15, 22, 17
        )

    with col2:

        address = st.selectbox(
            "Address",
            ["U", "R"]
        )

        famsize = st.selectbox(
            "Family Size",
            ["GT3", "LE3"]
        )

        Pstatus = st.selectbox(
            "Parents' Cohabitation Status",
            ["T", "A"]
        )

    with col3:

        Medu = st.selectbox(
            "Mother's Education",
            [0, 1, 2, 3, 4]
        )

        Fedu = st.selectbox(
            "Father's Education",
            [0, 1, 2, 3, 4]
        )

        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

    st.divider()

    # ======================================
    # FAMILY AND SOCIAL INFORMATION
    # ======================================

    st.subheader("2. Family and Social Information")

    col1, col2 = st.columns(2)

    with col1:

        Mjob = st.selectbox(
            "Mother's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        Fjob = st.selectbox(
            "Father's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        famrel = st.slider(
            "Family Relationship Quality",
            1, 5, 3
        )

        famsup = st.selectbox(
            "Family Educational Support",
            ["yes", "no"]
        )

        activities = st.selectbox(
            "Extra-curricular Activities",
            ["yes", "no"]
        )

    with col2:

        reason = st.selectbox(
            "Reason for Choosing School",
            ["home", "reputation", "course", "other"]
        )

        schoolsup = st.selectbox(
            "School Support",
            ["yes", "no"]
        )

        nursery = st.selectbox(
            "Attended Nursery School",
            ["yes", "no"]
        )

        romantic = st.selectbox(
            "In a Romantic Relationship",
            ["yes", "no"]
        )

        internet = st.selectbox(
            "Internet Access",
            ["yes", "no"]
        )

    st.divider()

    # ======================================
    # ACADEMIC INFORMATION
    # ======================================

    st.subheader("3. Academic Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        traveltime = st.selectbox(
            "Travel Time",
            [1, 2, 3, 4]
        )

        studytime = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4]
        )

        failures = st.number_input(
            "Past Class Failures",
            0, 4, 0
        )

    with col2:

        paid = st.selectbox(
            "Extra Paid Classes",
            ["yes", "no"]
        )

        higher = st.selectbox(
            "Wants Higher Education",
            ["yes", "no"]
        )

        absences = st.number_input(
            "Number of Absences",
            0, 100, 0
        )

    with col3:

        G1 = st.number_input(
            "First Period Grade (G1)",
            0, 20, 10
        )

        G2 = st.number_input(
            "Second Period Grade (G2)",
            0, 20, 10
        )

    st.divider()

    # ======================================
    # LIFESTYLE INFORMATION
    # ======================================

    st.subheader("4. Lifestyle Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        freetime = st.slider(
            "Free Time",
            1, 5, 3
        )

        goout = st.slider(
            "Going Out with Friends",
            1, 5, 3
        )

    with col2:

        Dalc = st.slider(
            "Workday Alcohol Consumption",
            1, 5, 1
        )

        Walc = st.slider(
            "Weekend Alcohol Consumption",
            1, 5, 1
        )

    with col3:

        health = st.slider(
            "Current Health Status",
            1, 5, 3
        )

    # ======================================
    # SUBMIT BUTTON
    # ======================================

    submit = st.form_submit_button(
        "🔍 Compare DNN and Random Forest Predictions",
        use_container_width=True
    )


# ==========================================
# PHASE 17: DNN AND RANDOM FOREST PREDICTION
# ==========================================

if submit:

    # ======================================
    # COLLECT STUDENT INPUTS
    # ======================================

    student_data = {

        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2

    }

    # ======================================
    # PREPROCESS STUDENT INFORMATION
    # ======================================

    input_df = pd.DataFrame([student_data])

    processed_input = preprocessor.transform(input_df)

    if hasattr(processed_input, "toarray"):

        processed_input = processed_input.toarray()

    processed_input = np.asarray(
        processed_input,
        dtype=np.float32
    )

    # ======================================
    # TARGET MAPPING
    # ======================================

    reverse_mapping = {

        value: key

        for key, value in target_mapping.items()

    }

    # ======================================
    # RANDOM FOREST PREDICTION
    # ======================================

    rf_prediction = rf_model.predict(
        processed_input
    )[0]

    rf_category = reverse_mapping[rf_prediction]

    rf_probabilities = rf_model.predict_proba(
        processed_input
    )[0]

    rf_class_index = list(
        rf_model.classes_
    ).index(rf_prediction)

    rf_confidence = (
        rf_probabilities[rf_class_index] * 100
    )

    # ======================================
    # DNN PREDICTION
    # ======================================

    dnn_probabilities = dnn_model.predict(
        processed_input,
        verbose=0
    )[0]

    dnn_prediction = int(
        np.argmax(dnn_probabilities)
    )

    dnn_category = reverse_mapping[dnn_prediction]

    dnn_confidence = (
        dnn_probabilities[dnn_prediction] * 100
    )

    # ======================================
    # DISPLAY PREDICTION RESULTS
    # ======================================

    st.divider()

    st.header("📊 Student Performance Prediction")

    st.markdown(
        "The same student information is evaluated by "
        "both trained models."
    )

    col1, col2 = st.columns(2)

    # ======================================
    # DNN RESULT
    # ======================================

    with col1:

        st.subheader("🧠 Deep Neural Network")

        st.metric(
            label="Predicted Performance",
            value=f"{dnn_category} Performance"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{dnn_confidence:.2f}%"
        )

        st.progress(
            float(dnn_confidence / 100)
        )

        if dnn_category == "High":

            st.success(
                "The DNN predicts High Academic Performance."
            )

        elif dnn_category == "Medium":

            st.warning(
                "The DNN predicts Medium Academic Performance."
            )

        else:

            st.error(
                "The DNN predicts Low Academic Performance."
            )

    # ======================================
    # RANDOM FOREST RESULT
    # ======================================

    with col2:

        st.subheader("🌳 Random Forest")

        st.metric(
            label="Predicted Performance",
            value=f"{rf_category} Performance"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{rf_confidence:.2f}%"
        )

        st.progress(
            float(rf_confidence / 100)
        )

        if rf_category == "High":

            st.success(
                "The Random Forest predicts High Academic Performance."
            )

        elif rf_category == "Medium":

            st.warning(
                "The Random Forest predicts Medium Academic Performance."
            )

        else:

            st.error(
                "The Random Forest predicts Low Academic Performance."
            )

    # ======================================
    # MODEL AGREEMENT COMPARISON
    # ======================================

    st.divider()

    st.subheader("🔍 Model Prediction Comparison")

    if dnn_category == rf_category:

        st.success(
            f"Both models agree! The predicted category is "
            f"{dnn_category} Academic Performance."
        )

    else:

        st.warning(
            "The two models predict different academic "
            "performance categories."
        )

        st.write(
            f"**DNN Prediction:** {dnn_category}"
        )

        st.write(
            f"**Random Forest Prediction:** {rf_category}"
        )

    # ======================================
    # OVERALL MODEL ACCURACY
    # ======================================

    st.divider()

    st.subheader("📈 Overall Model Accuracy Comparison")

    accuracy_col1, accuracy_col2 = st.columns(2)

    with accuracy_col1:

        st.metric(
            label="DNN Test Accuracy",
            value="76.53%"
        )

    with accuracy_col2:

        st.metric(
            label="Random Forest Test Accuracy",
            value="82.65%"
        )

    st.caption(
        "Test accuracy represents performance on the unseen "
        "test dataset. It is different from the confidence "
        "of an individual prediction."
    )

    # ======================================
    # FINAL EXPLANATION
    # ======================================

    st.divider()

    st.info(
        "Prediction confidence represents the probability "
        "estimated by each model for its predicted category. "
        "It is not a guarantee of actual academic performance."
    )
