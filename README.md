# Customer Churn Prediction API

## Overview
Live API for predicting customer churn risk using machine learning. Deployed FastAPI application with real-time predictions.

## API Endpoints
- `POST /predict/` - Returns JSON predictions
- `POST /predict_csv/` - Downloads predictions as CSV

## Data Format
Upload CSV with columns:
- CustomerID, AmountSpent, ProductCategory
- LoginFrequency, ServiceUsage  
- Days_Since_Last_Transaction
- Days_Since_last_Interaction
- Days_Since_Last_Login

## Live Demo
https://lloyds-bank-customer-churn.onrender.com/docs
