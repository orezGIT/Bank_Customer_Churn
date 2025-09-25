#%%

import pickle
import io
import csv
import numpy as np 
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from datetime import datetime

#%%

with open('stacking_model.pkl', 'rb') as f: 
    loaded_model = pickle.load(f)

app = FastAPI()   

MODEL_FEATURES = [
    'AmountSpent', 'ProductCategory', 'LoginFrequency',
    'ServiceUsage', 'Days_Since_Last_Transaction',
    'Days_Since_last_Interaction', 'Days_Since_Last_Login'
]


#%%
# create a simple root endpoint
@app.get("/")
def root():
    return {"message": "Customer churn prediction API is running."}

#%%

#Add a prediction endpoint to receive files and return predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)): 
    
    try: 
        # Read raw data
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Check required columns
        required_cols = ['CustomerID'] + MODEL_FEATURES
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing required columns: {missing_cols}"}
        
        # Separate CustomerID and features
        customer_id = df['CustomerID']
        X = df[MODEL_FEATURES]
                
        # Separate CustomerID and features
        customer_id = df['CustomerID']
        X = df.drop('CustomerID', axis=1)


        # Use your loaded model to predict 
        predictions = loaded_model.predict(X) 

        # Get probability of class 1 
        prediction_probibility = loaded_model.predict_proba(X)[:, 1]

        results = [
            {"CustomerID": str(cid), 
            "ChurnStatus": int(pred), 
            "ChurnProbability": round(float(proba), 3), 
            "Risk_Level": (
                "High" if proba > 0.7
                else "Medium" if proba > 0.4 
                else "low"
            )
            }
            for cid, pred, proba in zip(customer_id, predictions, prediction_probibility)
        ]

        return {"predictions": results}
    
    except Exception as e:
        return {"error": str(e)}  # Returns detailed error message

#Add a prediction endpoint to receive files and return CSV download
@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)): 
    try: 

        # Read raw data
        df = pd.read_csv(io.StringIO(await file.read()))

        # Check required columns
        required_cols = ['CustomerID'] + MODEL_FEATURES
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing required columns: {missing_cols}"}

        # Separate CustomerID and features
        customer_id = df['CustomerID']
        X = df.drop('CustomerID', axis=1)
        
        # Use your loaded model to predict 
        predictions = loaded_model.predict(X) 

        # Get probability of class 1
        prediction_probibility = loaded_model.predict_proba(X)[:, 1]

        results = [
            {"CustomerID": str(cid), 
            "ChurnStatus": int(pred), 
            "ChurnProbability": round(float(proba), 3), 
            "Risk_Level": (
                "High" if proba > 0.7
                else "Medium" if proba > 0.4 
                else "low"
            )
            }
            for cid, pred, proba in zip(customer_id, predictions, prediction_probibility)
        ]

        # create csv in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        output.seek(0)

        # Return the predictions as a list 
        return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})

    except Exception as e: 
        return {"error": str(e)}



