from supabase import create_client, Client
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests
import io

# Initialize FastAPI
app = FastAPI()

# Supabase Credentials
SUPABASE_URL = "https://klfiosrpujlpgsnxoqsg.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtsZmlvc3JwdWpscGdzbnhvcXNnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjIwODkzOCwiZXhwIjoyMDU3Nzg0OTM4fQ.WmxhcdloIcNPNoP5QHDfXvkWZ4rUoyIJew7N329lr7U"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Load Model from Supabase
MODEL_URL = "https://klfiosrpujlpgsnxoqsg.supabase.co/storage/v1/object/public/model/lead_scoring_model.pkl"
response = requests.get(MODEL_URL)

if response.status_code == 200:
    model = joblib.load(io.BytesIO(response.content))
    print("✅ Model loaded successfully from Supabase!")
else:
    raise Exception("❌ Failed to load model from Supabase")

# Define Input Data Schema
class InputData(BaseModel):
    feature1: float
    feature2: float

# Prediction Endpoint
@app.post("/predict/")
def predict(data: InputData):
    features = [[data.feature1, data.feature2]]
    prediction = model.predict(features)[0]

    # Save Prediction to Supabase Table
    prediction_data = {
        "feature1": data.feature1,
        "feature2": data.feature2,
        "prediction": prediction
    }

    response = supabase.table("predictions").insert(prediction_data).execute()

    if response.error:
        return {"error": f"Failed to store prediction: {response.error.message}"}

    return {"prediction": prediction}
