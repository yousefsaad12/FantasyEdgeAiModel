from flask import Flask, jsonify
import requests
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

# Global variables to store model and scaler
best_model = None
scaler = None
data = None
features = [
    "goalsScored", "assists", "cleanSheets", "penaltiesSaved", "penaltiesMissed",
    "ownGoals", "yellowCards", "redCards", "saves", "bonus", "bonusPointsSystem",
    "dreamTeamCount", "expectedGoals", "expectedAssists", "expectedGoalInvolvements",
    "expectedGoalsConceded", "expectedGoalsPer90", "expectedAssistsPer90",
    "goalsConcededPer90", "startsPer90", "cleanSheetsPer90"
]
target = "totalPoints"

# Train model function
def train_model():
    global best_model, scaler, data

    # Fetch data
    url = 'http://localhost:5235/api/player/playersdata/train'
    data = fetch_data(url)

    # Data preprocessing
    data["playerName"] = data["firstName"] + " " + data["secondName"]
    data = data.sort_values(by=["playerName", "gameWeek"])
    data["previousPoints"] = data.groupby("playerName")["totalPoints"].shift(1)
    data["avgPointsLast3"] = data.groupby("playerName")["totalPoints"].rolling(3).mean().reset_index(0, drop=True)
    data["maxPointsLast5"] = data.groupby("playerName")["totalPoints"].rolling(5).max().reset_index(0, drop=True)
    data = data.dropna(subset=["previousPoints", "avgPointsLast3", "maxPointsLast5"])

    X = data[features + ["avgPointsLast3", "maxPointsLast5"]]
    y = data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning and model training
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = xgb.XGBRegressor(objective="reg:squarederror", **best_params)
    best_model.fit(X_train, y_train)

    # Save the model to a file
    best_model.save_model("fantasy_edge_model.json")

    return "Model training completed!"

# Fetch data function
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# Endpoint to trigger training manually
@app.route('/train', methods=['GET'])
def train():
    return jsonify(message=train_model())

# Endpoint to make a prediction
@app.route('/predict/<player_name>', methods=['GET'])
def predict(player_name):
    result = predict_player(player_name)
    return jsonify(result)

# Predict player function (as defined earlier)
def predict_player(player_name):
    player_data = data[data["playerName"] == player_name]
    if player_data.empty:
        return f"Player '{player_name}' not found in the dataset."

    player_features = player_data[features + ["avgPointsLast3", "maxPointsLast5"]].iloc[-1:]
    player_features_scaled = scaler.transform(player_features)
    
    predicted_points = best_model.predict(player_features_scaled)[0]
    previous_points = player_data["previousPoints"].iloc[-1]
    
    if previous_points != 0:
        percentage_change = ((predicted_points - previous_points) / previous_points) * 100
    else:
        percentage_change = 0
    
    trend = "Increasing" if percentage_change > 0 else "Decreasing"
    position = player_data["position"].values[0]
    is_goalkeeper = position == 1

    if not is_goalkeeper:
        assists_percentage = player_data["assists"].iloc[-1] / player_data["totalPoints"].iloc[-1] * 100
        goals_percentage = player_data["goalsScored"].iloc[-1] / player_data["totalPoints"].iloc[-1] * 100
    else:
        clean_sheet_percentage = player_data["cleanSheets"].iloc[-1] / player_data["totalPoints"].iloc[-1] * 100

    result = {
        "playerName": player_name,
        "predictedPoints": round(predicted_points, 2),
        "percentageChange": f"{round(percentage_change, 2)}%",
        "trend": trend
    }

    if not is_goalkeeper:
        result["assistsPercentage"] = f"{round(assists_percentage, 2)}%"
        result["goalsPercentage"] = f"{round(goals_percentage, 2)}%"
    else:
        result["cleanSheetPercentage"] = f"{round(clean_sheet_percentage, 2)}%"

    return result

if __name__ == '__main__':
    app.run(debug=True)
