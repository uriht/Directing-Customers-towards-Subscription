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

# Create the Streamlit web app
st.title('FineTech App Enrollment Predictor')

# Upload widget for the Excel file in the main content area
uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded Excel file into a DataFrame
    user_data = pd.read_excel(uploaded_file)

    # Display the uploaded data
    st.subheader('Uploaded Data')
    st.write(user_data)

    # When the user clicks the prediction button
    if st.button('Predict Enrollment'):
        # Make sure the uploaded data matches the expected input features
        # Here, we assume that the columns in the uploaded data match the 48 features

        # Make a prediction
        prediction = predict_enrollment(user_data)

        # Display the prediction
        if prediction[0] == 1:
            st.success('User is likely to enroll.')
        else:
            st.error('User is unlikely to enroll.')

    # You can also add more information or visualizations based on the predictions if needed.
    # For example, display a bar chart of feature importance or a confusion matrix.

# Optionally, you can add explanations, explanations, and visualizations to help users understand the predictions.

# Example: Display feature importance
# st.subheader('Feature Importance')
# feature_importance = model.feature_importances_
# feature_names = X_train.columns
# df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
# st.bar_chart(df_importance.sort_values(by='Importance', ascending=False))

# Example: Display confusion matrix
# st.subheader('Confusion Matrix')
# st.write(confusion_matrix(y_test, y_pred_jl))
