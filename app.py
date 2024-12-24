import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load('titanic_model.pkl')

# Page Title
st.markdown("<h1 style='text-align: center; color: navy;'>Titanic Survival Prediction 🚢</h1>", unsafe_allow_html=True)

# Sidebar for Input
st.sidebar.header("Passenger Details")
Pclass = st.sidebar.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
Sex_male = st.sidebar.selectbox("Sex", ["male", "female"])
Age = st.sidebar.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
Sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.sidebar.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)

# Encode Inputs
Sex_male = 1 if Sex_male == "male" else 0
input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex_male': [Sex_male],
    'Age': [Age],
    'SibSp': [Sibsp],
    'Parch': [Parch],
    'Fare': [Fare],
})

# Reorder columns to match model requirements
input_data = input_data[model.feature_names_in_]

# Prediction
if st.button("Predict Survival"):
    probabilities = model.predict_proba(input_data)
    survival_probability = probabilities[0][1]

    # Output Result
    if survival_probability > 0.5:
        st.success(f" Survived! Probability: {survival_probability:.2f}")
    else:
        st.error(f"Did Not Survive. Probability: {survival_probability:.2f}")

    # Visualize Probability
    fig, ax = plt.subplots()
    ax.bar(["Did Not Survive", "Survived"], [1-survival_probability, survival_probability], color=['red', 'green'])
    ax.set_ylim(0, 1)
    st.pyplot(fig)


# Know More Button
if st.button("Know More"):
    # Open the HTML file in a browser tab
    st.markdown('<a href="know_more_titanic.html" target="_blank">Open Know More</a>', unsafe_allow_html=True)
