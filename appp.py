import joblib
import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Load the trained model using joblib
model = joblib.load("demo.pkl")

def main():
    st.markdown("<h1 style='color: red; font-size: 53px;'>Fuel Efficiency Prediction App</h1>", unsafe_allow_html=True)
    st.sidebar.header("User Input Features")
    st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://static.vecteezy.com/system/resources/previews/009/398/852/original/fuel-gauge-clipart-design-illustration-free-png.png" 
        width="400">
    </div>
    """,
    unsafe_allow_html=True,
)# Sidebar title
st.sidebar.markdown("<p class='sidebar-title'>User Input Features</p>", unsafe_allow_html=True)

# Sidebar sections with labeled inputs
with st.sidebar.expander("Engine Specs"):
    cylinders = st.slider("Cylinders", min_value=3, max_value=8, value=4)
    displacement = st.number_input("Displacement", min_value=50, max_value=500, value=200)
    horsepower = st.number_input("Horsepower", min_value=50, max_value=400, value=150)

with st.sidebar.expander("Vehicle Specs"):
    weight = st.number_input("Weight", min_value=1000, max_value=7000, value=3000)
    acceleration = st.number_input("Acceleration", min_value=5, max_value=25, value=15)
    model_year = st.slider("Model Year", min_value=70, max_value=82, value=75)

with st.sidebar.expander("Origin"):
    origin = st.selectbox("Select Origin", ["USA", "Europe", "Japan"])
    if origin == "USA":
        origin = 0
    elif origin == "Europe":
        origin = 1
    else:
        origin = 2
        
    # Create a dictionary to store user inputs
    user_input = [
        cylinders,
        displacement,
        horsepower,
        weight,
        acceleration,
        model_year,
        origin,
    ]

    # Create a DataFrame with user inputs for display
    input_df = pd.DataFrame([user_input])

    # Predict fuel efficiency when the user clicks the "Predict" button
    if st.sidebar.button("Predict"):
        try:
            if model is not None:
                prediction = model.predict(input_df)[0]
                st.sidebar.success(f"Predicted Fuel Efficiency: {prediction:.2f} mpg")
            else:
                st.sidebar.warning("Model is not loaded.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Display user input data
    st.subheader("User Input Data")
    st.write(input_df)

if __name__ == "__main__":
    main()



