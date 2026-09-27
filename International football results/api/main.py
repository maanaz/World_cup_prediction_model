from fastapi import FastAPI
from pathlib import Path
import joblib
from fastapi import HTTPException
import pandas as pd
from pydantic import BaseModel
import math
import numpy as np

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="FIFA World Cup Prediction API",
    description="Machine Learning API for FIFA World Cup match prediction",
    version="1.0.0"
)
class PredictionRequest(BaseModel):

    round: str
    match_index: int
    model: str
class ScorePredictionRequest(BaseModel):

    round: str
    match_index: int

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# LOAD MODELS
# ============================================================

lr_model = joblib.load(
    MODEL_DIR / "logistic_model.pkl"
)

rf_model = joblib.load(
    MODEL_DIR / "random_forest_model.pkl"
)

xgb_model = joblib.load(
    MODEL_DIR / "xgboost_model.pkl"
)

result_scaler = joblib.load(
    MODEL_DIR / "result_scaler.pkl"
)

home_goal_model = joblib.load(
    MODEL_DIR / "home_goal_model.pkl"
)

away_goal_model = joblib.load(
    MODEL_DIR / "away_goal_model.pkl"
)

score_scaler = joblib.load(
    MODEL_DIR / "score_scaler.pkl"
)


# ============================================================
# MODEL DICTIONARY
# ============================================================

MODELS = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {
    0: "A",
    1: "D",
    2: "H"
}

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "home_scored",
    "home_conceded",
    "home_win_rate",
    "away_scored",
    "away_conceded",
    "away_win_rate",
    "neutral",
    "tournament_weight",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "elo_home",
    "elo_away",
    "elo_diff",
    "home_draw_rate",
    "away_draw_rate",
    "home_goal_diff",
    "away_goal_diff"
]


LABEL_MAP = {
    0: "A",
    1: "D",
    2: "H"
}


RESULT_NAMES = {
    "A": "Away Win",
    "D": "Draw",
    "H": "Home Win"
}
def predict_with_model(model_name, feature_row):

    X = feature_row[FEATURES]

    model = MODELS[model_name]

    if model_name == "Logistic Regression":

        X_input = result_scaler.transform(X)

    else:

        X_input = X

    probabilities = model.predict_proba(X_input)[0]

    prediction = model.predict(X_input)[0]

    if model_name == "XGBoost":
        prediction = LABEL_MAP[int(prediction)]

    return prediction, probabilities

# ============================================================
# POISSON PROBABILITY
# ============================================================

def poisson_probability(goals, expected_goals):

    return (
        np.exp(-expected_goals)
        * expected_goals ** goals
        / math.factorial(goals)
    )
# ============================================================
# SCORE PREDICTION
# ============================================================

def predict_score(feature_row):

    X = feature_row[FEATURES]

    X_scaled = score_scaler.transform(X)

    home_lambda = home_goal_model.predict(
        X_scaled
    )[0]

    away_lambda = away_goal_model.predict(
        X_scaled
    )[0]

    home_lambda = max(
        float(home_lambda),
        0.01
    )

    away_lambda = max(
        float(away_lambda),
        0.01
    )

    scores = []

    for home_goals in range(0, 7):

        for away_goals in range(0, 7):

            probability = (
                poisson_probability(
                    home_goals,
                    home_lambda
                )
                *
                poisson_probability(
                    away_goals,
                    away_lambda
                )
            )

            scores.append(
                {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "probability": probability
                }
            )

    score_df = pd.DataFrame(scores)

    score_df["probability"] *= (
        1 / score_df["probability"].sum()
    )

    score_df = score_df.sort_values(
        "probability",
        ascending=False
    )

    return (
        home_lambda,
        away_lambda,
        score_df
    )
# ============================================================
# ROUTES
# ============================================================


@app.get("/")
def root():

    return {
        "message": "FIFA World Cup Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.get("/models")
def get_models():

    return {
        "models": list(MODELS.keys())
    }

@app.post("/predict")
def predict(request: PredictionRequest):

    round_files = {
        "Group Stage": "wc_group_stage_features.csv",
        "Round of 32": "r32_features.csv",
        "Round of 16": "r16_features.csv",
        "Quarter Finals": "qf_features.csv",
        "Semi Finals": "semi_features.csv",
        "Final": "final_features.csv"
    }

    if request.round not in round_files:

        raise HTTPException(
            status_code=400,
            detail="Invalid tournament round"
        )

    if request.model not in MODELS:

        raise HTTPException(
            status_code=400,
            detail="Invalid model name"
        )

    csv_path = BASE_DIR / round_files[request.round]

    df = pd.read_csv(csv_path)

    if request.match_index < 0 or request.match_index >= len(df):

        raise HTTPException(
            status_code=400,
            detail="Invalid match index"
        )

    row = df.iloc[request.match_index]

    feature_row = pd.DataFrame([row])

    prediction, probabilities = predict_with_model(
        request.model,
        feature_row
    )

    return {
        "round": request.round,
        "match_index": request.match_index,
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "model": request.model,
        "prediction": prediction,
        "result": RESULT_NAMES[prediction],
        "probabilities": {
            "home": float(probabilities[2]),
            "draw": float(probabilities[1]),
            "away": float(probabilities[0])
        }
    }
# ============================================================
# TOURNAMENT DATA
# ============================================================

ROUND_FILES = {
    "Group Stage": "wc_group_stage_features.csv",
    "Round of 32": "r32_features.csv",
    "Round of 16": "r16_features.csv",
    "Quarter Finals": "qf_features.csv",
    "Semi Finals": "semi_features.csv",
    "Final": "final_features.csv"
}


# ============================================================
# GET AVAILABLE ROUNDS
# ============================================================

@app.get("/rounds")
def get_rounds():

    return {
        "rounds": list(ROUND_FILES.keys())
    }


# ============================================================
# GET MATCHES FOR A ROUND
# ============================================================

@app.get("/matches/{round_name}")
def get_matches(round_name: str):

    if round_name not in ROUND_FILES:

        raise HTTPException(
            status_code=404,
            detail="Tournament round not found"
        )

    csv_path = BASE_DIR / ROUND_FILES[round_name]

    df = pd.read_csv(csv_path)

    matches = []

    for index, row in df.iterrows():

        matches.append(
            {
                "match_index": int(index),
                "home_team": row["home_team"],
                "away_team": row["away_team"]
            }
        )

    return {
        "round": round_name,
        "matches": matches
    }
@app.post("/predict/score")
def predict_score_endpoint(
    request: ScorePredictionRequest
):

    if request.round not in ROUND_FILES:

        raise HTTPException(
            status_code=400,
            detail="Invalid tournament round"
        )

    csv_path = BASE_DIR / ROUND_FILES[request.round]

    df = pd.read_csv(csv_path)

    if (
        request.match_index < 0
        or request.match_index >= len(df)
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid match index"
        )

    row = df.iloc[request.match_index]

    feature_row = pd.DataFrame([row])

    home_lambda, away_lambda, score_df = (
        predict_score(feature_row)
    )

    best_score = score_df.iloc[0]

    top_scores = score_df.head(10)

    return {
        "round": request.round,
        "match_index": request.match_index,
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "expected_goals": {
            "home": home_lambda,
            "away": away_lambda
        },
        "most_likely_score": {
            "home_goals": int(
                best_score["home_goals"]
            ),
            "away_goals": int(
                best_score["away_goals"]
            ),
            "probability": float(
                best_score["probability"]
            )
        },
        "top_scores": [
            {
                "home_goals": int(score["home_goals"]),
                "away_goals": int(score["away_goals"]),
                "probability": float(score["probability"])
            }
            for _, score in top_scores.iterrows()
        ]
    }