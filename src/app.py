import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(page_title="Crop Recommendation", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.header("Crop Recommendation")

# upload the file

input_method = st.selectbox("Select a method of data input", ["Manual Entry", "File Input"])

if input_method == "Manual Entry":
  st.warning("Enter all values to continue")
  N = st.text_input("Enter a value for N")
  P = st.text_input("Enter a value for P")
  temperature = st.text_input("Enter a value for temperature")
  humidity = st.text_input("Enter a value for Humidity")
  ph = st.text_input("Enter a value for ph")
  rainfall = st.text_input("Enter a value for rainfall")

  button_single = st.button("Get Predictions")

  if N and P and temperature and humidity and ph and rainfall and button_single:
    with st.spinner("Getting you the recommendations.."):
      
      data = {
        "N" : float(N),
        "P" : float(P),
        "temperature" : float(temperature),
        "humidity" : float(humidity),
        "ph" : float(ph),
        "rainfall" : float(rainfall)
      }
      
      predictions = requests.post("https://fastapi-backend-crop.azurewebsites.net/predict-single",json=data)

      preds_json = predictions.json()

      st.subheader("Recommendations")

      # st.write(preds_json["predictions"])

      df = pd.read_json(preds_json["predictions"])

      st.write(df)



if input_method == "File Input":
  st.warning("Upload only CSV files")
  prediction_file = st.file_uploader("Upload a file for prediction", type=["csv"])

  button = st.button("Upload file for prediction")
  if prediction_file is not None and button:
    preds_df = pd.read_csv(prediction_file)

    preds_df_json = preds_df.to_dict(orient="split")

    predictions = requests.post("https://fastapi-backend-crop.azurewebsites.net/predict-file",json=preds_df_json)

    preds_json = predictions.json()

    st.subheader("Recommendations")

    preds = preds_json["predictions"]


    df = pd.read_json(preds)

    st.write(df)

  


  
  



