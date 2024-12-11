from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import predict_player
from retrain import retrain_model
import pandas as pd
from dotenv import load_dotenv  # Import dotenv to load environment variables
import os
import joblib
# Load environment variables from .env
load_dotenv()

app = FastAPI()
best_model = joblib.load("fantasy_edge_model.pkl")
scaler = joblib.load("scaler.pkl")
# Global variables
data = pd.read_csv("Player_Data.csv")  # Load the data from the CSV file
features = [
    "goalsScored", "assists", "cleanSheets", "penaltiesSaved", "penaltiesMissed",
    "ownGoals", "yellowCards", "redCards", "saves", "bonus", "bonusPointsSystem",
    "dreamTeamCount", "expectedGoals", "expectedAssists", "expectedGoalInvolvements",
    "expectedGoalsConceded", "expectedGoalsPer90", "expectedAssistsPer90",
    "goalsConcededPer90", "startsPer90", "cleanSheetsPer90"
]

class PlayerNameRequest(BaseModel):
    player_name: str

@app.post("/predict/")
def predict_player_endpoint(request: PlayerNameRequest):
    # Call the `predict_player` function
    result = predict_player(request.player_name, data, features, best_model, scaler)
    
    if isinstance(result, str):  # If the result is an error message (player not found)
        raise HTTPException(status_code=404, detail=result)
    
    return result

@app.get("/retrain/")
def retrain_endpoint():
    # Read the API_URL from the environment
    api_url = os.getenv("API_URL")
    if not api_url:
        raise HTTPException(status_code=400, detail="API_URL not found in environment variables")

    # Call the retrain_model function with the URL from the .env file
    result = retrain_model(api_url)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result
