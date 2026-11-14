import streamlit as st
import pandas as pd
import numpy as np
import joblib

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

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    target_mapping = joblib.load(
        "models/target_mapping.pkl"
    )

    return rf_model, preprocessor, target_mapping


rf_model, preprocessor, target_mapping = load_models()

# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🎓 Student Academic Performance Predictor")

st.markdown(
    """
    Predict student academic performance using
    Machine Learning and Deep Learning.

    Enter the student's information below to
    predict their academic performance category.
    """
)

st.divider()

st.success("Machine Learning model loaded successfully!")


# ==========================================
# PHASE 16: STUDENT INPUT FORM
# ==========================================

st.header("📝 Enter Student Information")

st.write(
    "Fill in the details below to predict academic performance."
)

with st.form("student_form"):

    # PERSONAL INFORMATION
    st.subheader("1. Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Gender", ["F", "M"])
        age = st.number_input("Age", 15, 22, 17)

    with col2:
        address = st.selectbox("Address", ["U", "R"])
        famsize = st.selectbox("Family Size", ["GT3", "LE3"])
        Pstatus = st.selectbox("Parents' Cohabitation Status", ["T", "A"])

    with col3:
        Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4])
        Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4])
        guardian = st.selectbox(
            "Guardian", ["mother", "father", "other"]
        )

    st.divider()

    # FAMILY INFORMATION
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

        famrel = st.slider("Family Relationship Quality", 1, 5, 3)

        famsup = st.selectbox("Family Educational Support", ["yes", "no"])

        activities = st.selectbox("Extra-curricular Activities", ["yes", "no"])

    with col2:
        reason = st.selectbox(
            "Reason for Choosing School",
            ["home", "reputation", "course", "other"]
        )

        schoolsup = st.selectbox("School Support", ["yes", "no"])

        nursery = st.selectbox("Attended Nursery School", ["yes", "no"])

        romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])

        internet = st.selectbox("Internet Access", ["yes", "no"])

    st.divider()

    # ACADEMIC INFORMATION
    st.subheader("3. Academic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        traveltime = st.selectbox("Travel Time", [1, 2, 3, 4])
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
        failures = st.number_input("Past Class Failures", 0, 4, 0)

    with col2:
        paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
        higher = st.selectbox("Wants Higher Education", ["yes", "no"])
        absences = st.number_input("Number of Absences", 0, 100, 0)

    with col3:
        G1 = st.number_input("First Period Grade (G1)", 0, 20, 10)
        G2 = st.number_input("Second Period Grade (G2)", 0, 20, 10)

    st.divider()

    # LIFESTYLE INFORMATION
    st.subheader("4. Lifestyle Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        freetime = st.slider("Free Time", 1, 5, 3)
        goout = st.slider("Going Out with Friends", 1, 5, 3)

    with col2:
        Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)

    with col3:
        health = st.slider("Current Health Status", 1, 5, 3)

    # SUBMIT BUTTON
    submit = st.form_submit_button(
        "🔍 Predict Student Performance",
        use_container_width=True
    )


    # ==========================================
# PHASE 17: STUDENT PERFORMANCE PREDICTION
# ==========================================

if submit:

    # Collect all student inputs
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

    # Convert input into DataFrame
    input_df = pd.DataFrame([student_data])

    # Preprocess student information
    processed_input = preprocessor.transform(input_df)

    # Convert sparse matrix to dense if necessary
    if hasattr(processed_input, "toarray"):
        processed_input = processed_input.toarray()

    processed_input = np.asarray(
        processed_input,
        dtype=np.float32
    )

    # Predict performance
    prediction = rf_model.predict(processed_input)[0]

    # Convert numerical prediction into category
    reverse_mapping = {
        value: key
        for key, value in target_mapping.items()
    }

    predicted_category = reverse_mapping[prediction]

    # Get prediction confidence
    probabilities = rf_model.predict_proba(processed_input)[0]

    class_index = list(rf_model.classes_).index(prediction)

    confidence = probabilities[class_index] * 100

    # Display results
    st.divider()

    st.header("📊 Student Performance Prediction")

    st.subheader("Prediction Result")

    if predicted_category == "High":

        st.success("🟢 High Academic Performance")

        st.write(
            "The model predicts that this student "
            "is likely to achieve High Academic Performance."
        )

    elif predicted_category == "Medium":

        st.warning("🟡 Medium Academic Performance")

        st.write(
            "The model predicts that this student "
            "is likely to achieve Medium Academic Performance."
        )

    else:

        st.error("🔴 Low Academic Performance")

        st.write(
            "The model predicts that this student "
            "is likely to achieve Low Academic Performance."
        )

    # Display confidence
    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.2f}%"
    )

    st.progress(float(confidence / 100))

    st.caption(
        "Confidence represents the Random Forest model's "
        "estimated probability for the predicted category. "
        "It is not a guarantee of actual academic performance."
    )