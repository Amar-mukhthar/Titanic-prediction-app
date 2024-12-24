import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load('titanic_model.pkl')

# Page Title
st.markdown("<h1 style='text-align: center; color: navy;'>Titanic Survival Prediction 🚢</h1>", unsafe_allow_html=True)


# Add an introductory explanation
st.write("""
This app predicts whether a Titanic passenger would have survived or not, based on various details about the passenger. 
Enter the details in the sidebar to get the survival probability and an easy-to-understand result.
""")


# Sidebar for Input
st.sidebar.header("Passenger Details")
st.sidebar.write("Provide the following details to predict the survival:")

Pclass = st.sidebar.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
Sex_male = st.sidebar.selectbox("Sex", ["male", "female"])
Age = st.sidebar.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
Sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0,help="How many siblings or spouses did the passenger travel with? Enter 0 if they traveled alone.")
Parch = st.sidebar.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0,    help="How many parents or children did the passenger travel with? Enter 0 if none.")
Fare = st.sidebar.number_input("Fare (Ticket Price in USD)", min_value=0.0, max_value=1000.0, value=32.0)

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

        # Explain the result in simple terms
    st.write("""
    The prediction is based on historical data from the Titanic disaster. A probability above 50% indicates the passenger
    is likely to survive, while a lower probability suggests they might not survive. Factors like age, gender, ticket class,
    and whether they traveled with family impact the survival chances.
    """)

    # Visualize Probability
    fig, ax = plt.subplots()
    ax.bar(["Did Not Survive", "Survived"], [1-survival_probability, survival_probability], color=['red', 'green'])
    ax.set_ylim(0, 1)
    st.pyplot(fig)


if st.button("Know More"):
    with open("know_more_titanic.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    st.markdown(html_content, unsafe_allow_html=True)

