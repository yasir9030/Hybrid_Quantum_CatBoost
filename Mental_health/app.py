import streamlit as st
import numpy as np
import pickle
import pennylane as qml
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Hybrid Quantum–CatBoost Depression Detection",
    page_icon="🧠",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#F8FAFC;
}

h1{
    color:#0F172A;
    text-align:center;
}

.block-container{
    padding-top:2rem;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:18px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.title("🧠 Hybrid Quantum–CatBoost Depression Detection")

st.markdown("""
This application predicts **Student Depression**
using a **Hybrid Quantum Feature Extractor**
and **CatBoost Classifier**.
""")

st.divider()

# ==========================================================

import os
import pickle
import streamlit as st

@st.cache_resource
def load_model():

    model_path = os.path.join(
        "Mental_health",
        "Quantum_CatBoost_Model.sav"
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()
# ==========================================================
# QUANTUM DEVICE
# ==========================================================

n_qubits = 8

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)

# ==========================================================
# QUANTUM FEATURE MAP
# ==========================================================

@qml.qnode(dev)
def quantum_feature_map(x):

    qml.AngleEmbedding(
        features=x,
        wires=range(n_qubits),
        rotation="Y"
    )

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(n_qubits)
    ]

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def get_quantum_features(sample):

    sample = np.asarray(sample).reshape(-1)

    q_features = quantum_feature_map(sample)

    return np.array(q_features).reshape(1,-1)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Hybrid Quantum–CatBoost")

st.sidebar.info(
"""
Student Depression Detection

Model : Hybrid Quantum + CatBoost

Qubits : 8

Quantum Encoding : AngleEmbedding

Backend : PennyLane (GPU)
"""
)

# ==========================================================
# INPUT SECTION
# ==========================================================

st.header("Enter Student Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=60,
        value=22
    )

    academic_pressure = st.slider(
        "Academic Pressure",
        0,
        5,
        3
    )

    study_satisfaction = st.slider(
        "Study Satisfaction",
        0,
        5,
        3
    )

    degree_encoder = LabelEncoder()
    degree_encoder.classes_ = np.array([
        'B.Arch', 'B.Com', 'B.Ed', 'B.Pharm', 'B.Tech',
        'BA', 'BBA', 'BCA', 'BE', 'BHM',
        'BSc', 'Class 12', 'LLB', 'LLM',
        'M.Com', 'M.Ed', 'M.Pharm', 'M.Tech',
        'MA', 'MBA', 'MBBS', 'MCA', 'MD',
        'ME', 'MHM', 'MSc', 'Others', 'PhD'
    ])

    degree = st.selectbox(
        "Degree",
        degree_encoder.classes_
    )

    degree_encoded = degree_encoder.transform([degree])[0]


  
with col2:

    diet_options = [
        "Healthy",
        "Moderate",
        "Others",
        "Unhealthy"
    ]

    dietary_habits = st.selectbox(
        "Dietary Habits",
        diet_options
    )

    diet_encoded = diet_options.index(dietary_habits)

    suicidal_options = [
        "No",
        "Yes"
    ]

    suicidal = st.selectbox(
        "Have you ever had suicidal thoughts?",
        suicidal_options
    )

    suicidal_encoded = suicidal_options.index(suicidal)

    work_hours = st.slider(
        "Work/Study Hours",
        0,
        24,
        8
    )

    financial_stress = st.slider(
        "Financial Stress",
        0,
        5,
        2
    )


sample = np.array([[
    age,
    academic_pressure,
    study_satisfaction,
    degree_encoded,
    diet_encoded,
    suicidal_encoded,
    work_hours,
    financial_stress
]])

st.divider()

# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button("Predict Depression"):

    with st.spinner("Generating Quantum Features..."):

        Q_sample = get_quantum_features(sample)

        prediction = model.predict(Q_sample)

        probability = model.predict_proba(Q_sample)

    st.success("Prediction Completed")

    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 0:

        st.success("🟢 Not Depressed")

    else:

        st.error("🔴 Depressed")

    st.subheader("Prediction Probability")

    st.metric(
        "Not Depressed",
        f"{probability[0][0]*100:.2f}%"
    )

    st.metric(
        "Depressed",
        f"{probability[0][1]*100:.2f}%"
    )

    st.progress(float(probability[0][1]))

    # ==========================================================
# RESULT DASHBOARD
# ==========================================================

if st.button("Show Model Information"):

    st.markdown("## 📊 Hybrid Quantum Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
        """
        **Quantum Feature Extractor**

        • AngleEmbedding

        • 8 Qubits

        • PennyLane

        • GPU Simulator
        """
        )

    with col2:
        st.success(
        """
        **Classifier**

        • CatBoost

        • Binary Classification

        • Quantum Features

        • Fast Prediction
        """
        )

    with col3:
        st.warning(
        """
        **Selected Features**

        ✔ Age

        ✔ Academic Pressure

        ✔ Study Satisfaction

        ✔ Degree

        ✔ Dietary Habits

        ✔ Suicidal Thoughts

        ✔ Work/Study Hours

        ✔ Financial Stress
        """
        )

st.divider()

# ==========================================================
# MODEL DETAILS
# ==========================================================

