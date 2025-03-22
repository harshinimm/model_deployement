import joblib
import requests
import io
from fastapi import FastAPI
from pydantic import BaseModel

# Corrected Supabase URL (Remove extra slash)
MODEL_URL = "https://klfiosrpujlpgsnxoqsg.supabase.co/storage/v1/object/public/model/lead_scoring_model.pkl"

# Load Model from Supabase
response = requests.get(MODEL_URL)
if response.status_code == 200:
    model = joblib.load(io.BytesIO(response.content))
    print("✅ Model loaded successfully from Supabase!")
else:
    raise Exception("❌ Failed to load model from Supabase")

# Initialize FastAPI
app = FastAPI()

# Define Expected Input Data Structure
class InputData(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict/")
def predict(data: InputData):
    try:
        # Adjust Feature Input Shape if Needed
        features = [[data.feature1, data.feature2]]  # Ensure it matches model input shape
        
        # Make Prediction
        prediction = model.predict(features)
        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": str(e)}

# Run with: `uvicorn filename:app --reload`
