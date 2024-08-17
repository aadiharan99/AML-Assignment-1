import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Crop Recommendation", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.header("Crop Recommendation")

# upload the file

input_method = st.selectbox("Select a method of data input", ["Manual Entry", "File Input"])

if input_method == "Manual Entry":
  st.warning("Enter all values to continue")

if input_method == "File Input":
  st.warning("Upload only CSV files")
  prediction_file = st.file_uploader("Upload a file for prediction", type=["csv"])
  preds_df = pd.read_csv(prediction_file)

  st.write(preds_df)

  


  
  



