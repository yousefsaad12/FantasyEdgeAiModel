import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load model and scaler (ensure this is done only once)

def predict_player(player_name, data, features, best_model, scaler):
    # Filter player data by name
    player_data = data[data["playerName"] == player_name]
    if player_data.empty:
        return f"Player '{player_name}' not found in the dataset."

    # Prepare player features
    player_features = player_data[features + ["avgPointsLast3", "maxPointsLast5"]].iloc[-1:]  # Latest gameweek features
    player_features_scaled = scaler.transform(player_features)
    
    # Predict next gameweek points
    predicted_points = best_model.predict(player_features_scaled)[0]
    
    # Convert numpy.float32 to native Python float
    predicted_points = float(predicted_points)
    
    # Get previous points for the player (last gameweek)
    previous_points = player_data["previousPoints"].iloc[-1]
    
    # Convert numpy.float32 to native Python float
    previous_points = float(previous_points)
    
    # Calculate percentage change and trend
    if previous_points != 0:
        percentage_change = ((predicted_points - previous_points) / previous_points) * 100
    else:
        percentage_change = 0  # or another default value like 'N/A'
    
    # Convert percentage_change to float
    percentage_change = float(percentage_change)
    
    trend = "Increasing" if percentage_change > 0 else "Decreasing"

    # Check if the player is a goalkeeper (position 1)
    position = player_data["position"].values[0]
    
    # Check if the position is 1 (goalkeeper)
    is_goalkeeper = position == 1

    # Calculate percentages for non-goalkeepers
    if not is_goalkeeper:
        assists_percentage = player_data["assists"].iloc[-1] / player_data["totalPoints"].iloc[-1] * 100 if player_data["totalPoints"].iloc[-1] > 0 else 0
        goals_percentage = player_data["goalsScored"].iloc[-1] / player_data["totalPoints"].iloc[-1] * 100 if player_data["totalPoints"].iloc[-1] > 0 else 0
    else:
        clean_sheet_percentage = player_data["cleanSheets"].iloc[-1] / player_data["totalPoints"].iloc[-1] * 100 if player_data["totalPoints"].iloc[-1] > 0 else 0

    # Convert percentages to native Python float
    if not is_goalkeeper:
        assists_percentage = float(assists_percentage)
        goals_percentage = float(goals_percentage)
    else:
        clean_sheet_percentage = float(clean_sheet_percentage)

    # Build the result dictionary
    result = {
        "playerName": player_name,
        "predictedPoints": round(predicted_points, 2),
        "percentageChange": f"{round(percentage_change, 2)}%",
        "trend": trend
    }

    # Add statistics based on whether the player is a goalkeeper or not
    if not is_goalkeeper:
        result["assistsPercentage"] = f"{round(assists_percentage, 2)}%"
        result["goalsPercentage"] = f"{round(goals_percentage, 2)}%"
    else:
        result["cleanSheetPercentage"] = f"{round(clean_sheet_percentage, 2)}%"

    return result
