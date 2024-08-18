from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import logging
import pandas as pd
import uvicorn

# Load the model from the pickle file
with open('../models/svc_cross_val_tr_0.9_ts_0.1-3.pkl', 'rb') as file:
    model = pickle.load(file)

with open('../models/scaler_standard-2.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)

with open('../models/labels_mapping_dict_inverse.pkl', 'rb') as file:
    inverse_mapping = pickle.load(file)

print(inverse_mapping)

# Initialize FastAPI
app = FastAPI()


@app.post("/predict-single")
async def predict_single(data:dict):

    print(data)
    input_df = pd.DataFrame(data, index = [0])
    # print(input_df)
    # df_columns = [i for i in input_df.colum]
    # input_df = input_df[[df_columns]]
    input_df_copy = pd.DataFrame(standard_scaler.transform(input_df), columns=input_df.columns)
    print(input_df_copy)
    preds = model.predict(input_df.values)

    input_df["recommended_crops"] = preds

    print(input_df)

    # post-processing to get the correct name of the crops back
    
    input_df["recommended_crops"] = input_df["recommended_crops"].apply(lambda x: inverse_mapping[x])
    return {"predictions": input_df.to_json(orient="records")}


# Define the prediction endpoint
@app.post("/predict-file")
async def predict_file(data:dict):

    print("Hi :)")
    print(data)
    data_df = pd.DataFrame(data=data["data"], index=data["index"], columns=data["columns"])
    print(data_df)
    data_df_copy = data_df.copy()
    data_df = pd.DataFrame(standard_scaler.transform(data_df), columns=data_df.columns)
    data_df_copy["recommended_crops"] = data_df[[i for i in data_df.columns]].apply(
    lambda s: inverse_mapping[model.predict(s.values[None])[0]], axis=1
)
    
    print(data_df_copy.to_json(orient="records"))
    
    # Return the prediction result
    return {"predictions": data_df_copy.to_json(orient="records")}



if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')