import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

# 1. App Creation 

app = FastAPI(
    title="Indian Railways Price Prediction API",
    description="An API to predict train ticket base fares based on various features.",
    version="1.0.0"
)

origins = [
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)


# 2. Model Loading 

try:
    model_payload = joblib.load('price_prediction_model.pkl')
    model = model_payload['model']
    feature_order = model_payload['feature_order']
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'price_prediction_model.pkl' not found.")
    print("Please run the 'train_model.py' script first to train and save the model.")
    model = None
    feature_order = None

# 3. Input Schema Definition 

class TrainInput(BaseModel):
    class_code: Literal['1A', '2A', '3A', '3E', 'CC', 'SL', '2S', 'FC'] = Field(
        ...,
        description="The class of travel.",
        example="3A"
    )
    distance: int = Field(
        ...,
        gt=0,
        description="Total distance of the journey in kilometers.",
        example=1250
    )
    duration: int = Field(
        ...,
        gt=0,
        description="Total duration of the journey in minutes.",
        example=1500
    )
    has_catering: bool = Field(
        ...,
        description="Whether the train offers catering service.",
        example=True
    )
    is_dynamic: bool = Field(
        ...,
        description="Whether the train has dynamic pricing.",
        example=False
    )

class PredictionOut(BaseModel):
    predicted_base_fare: float = Field(
        ...,
        description="The predicted base fare for the ticket (excluding reservation and superfast charges)."
    )


# 4. API Endpoints 
@app.get("/", tags=["General"])
def read_root():

    return {"message": "Welcome to the Indian Railways Price Prediction API!"}

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
def predict_fare(payload: TrainInput):
    """
    Predicts the base fare of a train ticket.

    """
    if model is None or feature_order is None:
        return {"error": "Model is not loaded. Please check the server logs."}

    input_df = pd.DataFrame([payload.dict()])

    # Preprocessing 
    input_df.rename(columns={
        'has_catering': 'if_offering_catering',
        'is_dynamic': 'if_dynamic_fare'
    }, inplace=True)

    input_df['class_code'] = 'class_' + input_df['class_code']
    for col in feature_order:
        if col.startswith('class_'):
            input_df[col] = (input_df['class_code'] == col).astype(int)

    input_df.drop('class_code', axis=1, inplace=True)

    # Ensure all required feature columns are present and in the correct order
    final_df = pd.DataFrame(columns=feature_order)
    final_df = pd.concat([final_df, input_df])
    final_df.fillna(0, inplace=True)
    final_df = final_df[feature_order] 

    prediction = model.predict(final_df)

    return {"predicted_base_fare": round(prediction[0], 2)}

