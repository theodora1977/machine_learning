import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import os

app = FastAPI()

# Enable CORS to allow the frontend to connect to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and train the model on startup
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "gig_intl_prices_cleaned.csv")

try:
    data = pd.read_csv(csv_path)
    X = data[["KG", "Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6", "Zone 7"]]
    y = data["Zone 8"]
    
    model = LinearRegression()
    model.fit(X, y)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error loading data or training model: {e}")
    model = None
    data = None

# Define the input structure expected from the frontend
class PredictionRequest(BaseModel):
    kg: float
    zones: list[float]

@app.get("/")
def read_root():
    return FileResponse(os.path.join(current_dir, "index.html"))

@app.post("/predict")
def predict_zone_8(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    if len(request.zones) != 7:
        raise HTTPException(status_code=400, detail="Exactly 7 zone prices are required.")

    kg = request.kg
    zones = request.zones

    # Input Validation (Logic mirrored from predict.py)
    if kg <= 0:
        raise HTTPException(status_code=400, detail="KG must be positive")

    for i, val in enumerate(zones, 1):
        col = f"Zone {i}"
        # Check if the value is within the min-max range of the training data
        if col in data.columns:
            min_val = data[col].min()
            max_val = data[col].max()
            if not (min_val <= val <= max_val):
                raise HTTPException(status_code=400, detail=f"{col} price out of range ({min_val} - {max_val})")

    # Prepare input for prediction
    # The model expects a 2D array: [[kg, zone1, zone2, ... zone7]]
    input_features = np.array([[kg] + zones])
    
    try:
        predicted_price = model.predict(input_features)[0]
        return {"predicted_price": round(predicted_price, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    # Run the API on localhost port 5000
    # This will block until you press Ctrl+C
    print("Server is starting...")
    print("Open this link in your browser: http://localhost:5000")
    uvicorn.run(app, host="127.0.0.1", port=5000)