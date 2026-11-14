import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PHASE 1: PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# PHASE 2: LOAD TRAINED MODELS
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
# PHASE 3: APPLICATION HEADER
# ==========================================

st.title("🎓 Student Academic Performance Predictor")

st.markdown(
    """
    Welcome to the Student Academic Performance Prediction System.

    Enter student details and academic information to
    predict the student's academic performance using
    Machine Learning.
    """
)

st.divider()

st.success("Prediction model loaded successfully!")


# ==========================================
# PHASE 4: STUDENT INFORMATION FORM
# ==========================================

st.header("📝 Student Registration Form")

with st.form("student_form"):

    # --------------------------------------
    # 1. PERSONAL INFORMATION
    # --------------------------------------

    st.subheader("1. Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        student_name = st.text_input(
            "Student Full Name"
        )

        roll_number = st.text_input(
            "Roll Number"
        )

        student_class = st.selectbox(
            "Class",
            ["9", "10", "11", "12", "Other"]
        )

        section = st.selectbox(
            "Section",
            ["A", "B", "C", "D", "Other"]
        )

        email = st.text_input(
            "Email Address"
        )

    with col2:

        phone = st.text_input(
            "Phone Number"
        )

        full_address = st.text_area(
            "Full Address"
        )

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=22,
            value=17
        )

        sex = st.selectbox(
            "Gender",
            ["F", "M"],
            format_func=lambda x:
                "Female" if x == "F" else "Male"
        )

    school = st.selectbox(
        "School",
        ["GP", "MS"],
        format_func=lambda x:
            "GP - Gabriel Pereira" if x == "GP"
            else "MS - Mousinho da Silveira"
    )

    address = st.selectbox(
        "Residential Area",
        ["U", "R"],
        format_func=lambda x:
            "Urban" if x == "U" else "Rural"
    )

    st.divider()


    # --------------------------------------
    # 2. FAMILY INFORMATION
    # --------------------------------------

    st.subheader("2. Family Information")

    st.markdown("#### Father's Details")

    col1, col2 = st.columns(2)

    with col1:

        father_name = st.text_input(
            "Father's Name"
        )

        father_qualification = st.selectbox(
            "Father's Qualification",
            [
                "None",
                "Primary Education",
                "5th to 9th Grade",
                "Secondary Education",
                "Higher Education"
            ]
        )

    with col2:

        father_job = st.selectbox(
            "Father's Occupation",
            [
                "teacher",
                "health",
                "services",
                "at_home",
                "other"
            ]
        )

    st.markdown("#### Mother's Details")

    col1, col2 = st.columns(2)

    with col1:

        mother_name = st.text_input(
            "Mother's Name"
        )

        mother_qualification = st.selectbox(
            "Mother's Qualification",
            [
                "None",
                "Primary Education",
                "5th to 9th Grade",
                "Secondary Education",
                "Higher Education"
            ]
        )

    with col2:

        mother_job = st.selectbox(
            "Mother's Occupation",
            [
                "teacher",
                "health",
                "services",
                "at_home",
                "other"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        famsize = st.selectbox(
            "Family Size",
            ["GT3", "LE3"],
            format_func=lambda x:
                "More than 3 members"
                if x == "GT3"
                else "3 or fewer members"
        )

    with col2:

        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

    famsup = st.selectbox(
        "Family Educational Support",
        ["yes", "no"],
        format_func=lambda x:
            "Yes" if x == "yes" else "No"
    )

    st.divider()


    # --------------------------------------
    # 3. ACADEMIC INFORMATION
    # --------------------------------------

    st.subheader("3. Academic Information")

    col1, col2 = st.columns(2)

    with col1:

        G1 = st.number_input(
            "Previous Grade - G1",
            min_value=0,
            max_value=20,
            value=10
        )

        G2 = st.number_input(
            "Previous Grade - G2",
            min_value=0,
            max_value=20,
            value=10
        )

        studytime = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2 to 5 hours",
                3: "5 to 10 hours",
                4: "More than 10 hours"
            }[x]
        )

        failures = st.number_input(
            "Previous Class Failures",
            min_value=0,
            max_value=4,
            value=0
        )

    with col2:

        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=0
        )

        traveltime = st.selectbox(
            "Travel Time to School",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 15 minutes",
                2: "15 to 30 minutes",
                3: "30 minutes to 1 hour",
                4: "More than 1 hour"
            }[x]
        )

        paid = st.selectbox(
            "Extra Paid Classes",
            ["yes", "no"]
        )

        higher = st.selectbox(
            "Interested in Higher Education",
            ["yes", "no"]
        )

    st.divider()


    # --------------------------------------
    # 4. ADDITIONAL INFORMATION
    # --------------------------------------

    st.subheader("4. Additional Information")

    col1, col2 = st.columns(2)

    with col1:

        schoolsup = st.selectbox(
            "School Educational Support",
            ["yes", "no"]
        )

        activities = st.selectbox(
            "Participates in Extracurricular Activities",
            ["yes", "no"]
        )

    with col2:

        internet = st.selectbox(
            "Internet Access at Home",
            ["yes", "no"]
        )

        health = st.selectbox(
            "Health Status",
            [1, 2, 3, 4, 5],
            index=2
        )

    st.divider()

    # --------------------------------------
    # SUBMIT BUTTON
    # --------------------------------------

    submit = st.form_submit_button(
        "🔍 Predict Academic Performance",
        use_container_width=True
    )


# ==========================================
# PHASE 5: PREDICTION
# ==========================================

if submit:

    # Validate basic student details

    if not student_name.strip():

        st.error("Please enter the student's name.")

    elif not roll_number.strip():

        st.error("Please enter the roll number.")

    elif not email.strip() or "@" not in email:

        st.error("Please enter a valid email address.")

    else:

        # Convert qualifications into dataset values

        qualification_mapping = {
            "None": 0,
            "Primary Education": 1,
            "5th to 9th Grade": 2,
            "Secondary Education": 3,
            "Higher Education": 4
        }

        Medu = qualification_mapping[
            mother_qualification
        ]

        Fedu = qualification_mapping[
            father_qualification
        ]


        # ----------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------

        student_data = {

            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,

            # Fixed default for hidden feature
            "Pstatus": "T",

            "Medu": Medu,
            "Fedu": Fedu,

            "Mjob": mother_job,
            "Fjob": father_job,

            # Fixed defaults for hidden features
            "reason": "course",

            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,

            "schoolsup": schoolsup,
            "famsup": famsup,

            "paid": paid,
            "activities": activities,

            # Fixed defaults for hidden features
            "nursery": "yes",

            "higher": higher,
            "internet": internet,

            # Fixed defaults for hidden features
            "romantic": "no",
            "famrel": 3,
            "freetime": 3,
            "goout": 3,
            "Dalc": 1,
            "Walc": 1,

            "health": health,
            "absences": absences,
            "G1": G1,
            "G2": G2
        }


        # ----------------------------------
        # PREPROCESS INPUT
        # ----------------------------------

        input_df = pd.DataFrame(
            [student_data]
        )

        processed_input = preprocessor.transform(
            input_df
        )

        if hasattr(processed_input, "toarray"):

            processed_input = processed_input.toarray()

        processed_input = np.asarray(
            processed_input,
            dtype=np.float32
        )


        # ----------------------------------
        # GENERATE PREDICTION
        # ----------------------------------

        prediction = rf_model.predict(
            processed_input
        )[0]

        reverse_mapping = {
            value: key
            for key, value in target_mapping.items()
        }

        predicted_category = reverse_mapping[
            prediction
        ]

        probabilities = rf_model.predict_proba(
            processed_input
        )[0]

        class_index = list(
            rf_model.classes_
        ).index(prediction)

        confidence = (
            probabilities[class_index] * 100
        )


        # ==================================
        # PHASE 6: DISPLAY PREDICTION REPORT
        # ==================================

        st.divider()

        st.header("📊 Student Performance Report")

        st.subheader(
            f"Student: {student_name}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Roll Number",
                roll_number
            )

        with col2:

            st.metric(
                "Class",
                f"{student_class} - {section}"
            )

        with col3:

            st.metric(
                "Age",
                age
            )

        st.divider()

        st.subheader("Predicted Academic Performance")

        if predicted_category == "High":

            st.success(
                "🟢 HIGH ACADEMIC PERFORMANCE"
            )

            st.write(
                "The student is predicted to achieve "
                "High Academic Performance."
            )

        elif predicted_category == "Medium":

            st.warning(
                "🟡 MEDIUM ACADEMIC PERFORMANCE"
            )

            st.write(
                "The student is predicted to achieve "
                "Medium Academic Performance."
            )

        else:

            st.error(
                "🔴 LOW ACADEMIC PERFORMANCE"
            )

            st.write(
                "The student is predicted to achieve "
                "Low Academic Performance."
            )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            float(confidence / 100)
        )

        st.caption(
            "Prediction confidence is the model's estimated "
            "probability for the predicted category. "
            "It is not a guarantee of actual performance."
        )

        st.divider()

        st.subheader("Academic Details")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric("G1 Grade", G1)

        with col2:

            st.metric("G2 Grade", G2)

        with col3:

            st.metric("Absences", absences)

        st.info(
            "This prediction is generated using the "
            "trained Random Forest machine learning model."
        )