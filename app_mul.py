import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained XGBoost model
model = joblib.load('FineTech_app_ML_model.joblib')

# Define a function to predict enrollment
def predict_enrollment(data):
    # Make predictions using the loaded model
    predictions = model.predict(data)

    return predictions

# Create the Streamlit web app with decorations
st.set_page_config(
    page_title="FineTech Enrollment Predictor",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title with decoration
st.title('📈 FineTech App Enrollment Predictor 📊')

# Upload widget for the Excel file in the main content area
uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])

#Sample Dataset
@st.cache
def load_data(file_path):
    return pd.read_excel(file_path)

file_path = 'Testing_Director.xlsx'

data = load_data(file_path)

st.write(data)

if st.button('Download Excel File'):
    st.download_button(
        label='Download Data as Excel',
        data=data.to_excel(index=False).getvalue(),
        file_name='data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded Excel file into a DataFrame
    user_data = pd.read_excel(uploaded_file)

    # Display the uploaded data with a subheader
    st.subheader('📄 Uploaded Data 📄')
    st.write(user_data)

    # When the user clicks the prediction button
    if st.button('🔮 Predict Enrollment 🔮'):
        st.subheader('🚀 Predictions 🚀')
        
        # Initialize an empty list to store the results
        results = []

        # Iterate through each row of the user_data DataFrame
        for index, row in user_data.iterrows():
            # Make a prediction for the current user
            prediction = predict_enrollment(row.values.reshape(1, -1))

            # Append the result to the results list
            results.append({'User': index, 'Enrollment Prediction': 'Enroll' if prediction[0] == 1 else 'Not Enroll'})

        # Convert the results list to a DataFrame and display it as a table
        results_df = pd.DataFrame(results)
        st.write(results_df)
