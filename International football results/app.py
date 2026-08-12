import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import math
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="FIFA World Cup AI Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FIFA World Cup 2026")
st.subheader("Machine Learning Match Prediction & Football Analytics")


# ============================================================
# CONSTANTS
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


# ============================================================
# LOAD MODELS
# ============================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_models():

    lr = joblib.load(MODEL_DIR / "logistic_model.pkl")
    rf = joblib.load(MODEL_DIR / "random_forest_model.pkl")
    xgb = joblib.load(MODEL_DIR / "xgboost_model.pkl")

    result_scaler = joblib.load(MODEL_DIR / "result_scaler.pkl")

    home_goal_model = joblib.load(MODEL_DIR / "home_goal_model.pkl")
    away_goal_model = joblib.load(MODEL_DIR / "away_goal_model.pkl")
    score_scaler = joblib.load(MODEL_DIR / "score_scaler.pkl")

    return (
        lr,
        rf,
        xgb,
        result_scaler,
        home_goal_model,
        away_goal_model,
        score_scaler
    )

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

lr_model = joblib.load(MODEL_DIR / "logistic_model.pkl")
rf_model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
xgb_model = joblib.load(MODEL_DIR / "xgboost_model.pkl")

result_scaler = joblib.load(MODEL_DIR / "result_scaler.pkl")

home_goal_model = joblib.load(MODEL_DIR / "home_goal_model.pkl")
away_goal_model = joblib.load(MODEL_DIR / "away_goal_model.pkl")
score_scaler = joblib.load(MODEL_DIR / "score_scaler.pkl")

MODELS = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}


# ============================================================
# LOAD CSV
# ============================================================
@st.cache_data
def load_csv(filename):

    csv_path = BASE_DIR / filename

    df = pd.read_csv(csv_path)

    df = df.loc[
        :,
        ~df.columns.str.startswith("Unnamed:")
    ]

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    return df


# ============================================================
# ROUND FILES
# ============================================================

ROUND_FILES = {

    "Group Stage":
        "wc_group_stage_features.csv",

    "Round of 32":
        "r32_features.csv",

    "Round of 16":
        "r16_features.csv",

    "Quarter Finals":
        "qf_features.csv",

    "Semi Finals":
        "semi_features.csv",

    "Final":
        "final_features.csv"
}


# ============================================================
# LOAD ROUND
# ============================================================

def load_round(round_name):

    return load_csv(
        ROUND_FILES[round_name]
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_with_model(
    model_name,
    feature_row
):

    X = feature_row[FEATURES]

    model = MODELS[model_name]

    if model_name == "Logistic Regression":

        X_input = result_scaler.transform(X)

        probabilities = model.predict_proba(
            X_input
        )[0]

        prediction = model.predict(
            X_input
        )[0]

    else:

        X_input = X

        probabilities = model.predict_proba(
            X_input
        )[0]

        prediction = model.predict(
            X_input
        )[0]

    if model_name == "XGBoost":

        prediction = LABEL_MAP[int(prediction)]

    return prediction, probabilities


# ============================================================
# POISSON SCORE PREDICTION
# ============================================================

def poisson_probability(
    goals,
    expected_goals
):

    return (
        np.exp(-expected_goals)
        *
        expected_goals ** goals
        /
        math.factorial(goals)
    )


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
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Prediction Settings")

page = st.sidebar.radio(
    "Go to",
    [
        "🏆 Tournament Predictions",
        "⚽ Match Predictor",
        "📊 Model Evaluation"
    ]
)


# ============================================================
# TOURNAMENT PREDICTIONS
# ============================================================

if page == "🏆 Tournament Predictions":

    st.header("🏆 Tournament Center")

    st.markdown(
        "Explore how the machine learning models predict every stage "
        "of the 2026 FIFA World Cup."
    )

    st.divider()

    # ========================================================
    # CONTROLS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        round_name = st.selectbox(
            "🏟️ Tournament Round",
            list(ROUND_FILES.keys())
        )

    with col2:

        model_name = st.selectbox(
            "🤖 Prediction Model",
            [
                "XGBoost",
                "Random Forest",
                "Logistic Regression"
            ]
        )

    df = load_round(round_name)

    # ========================================================
    # PREDICT ALL MATCHES
    # ========================================================

    predictions = []

    for _, row in df.iterrows():

        feature_row = pd.DataFrame([row])

        prediction, probabilities = predict_with_model(
            model_name,
            feature_row
        )

        predictions.append(
            {
                "date": row["date"],
                "home": row["home_team"],
                "away": row["away_team"],
                "prediction": prediction,
                "home_prob": probabilities[2] * 100,
                "draw_prob": probabilities[1] * 100,
                "away_prob": probabilities[0] * 100
            }
        )

    pred_df = pd.DataFrame(predictions)

    # ========================================================
    # SUMMARY CARDS
    # ========================================================

    home_predictions = (
        pred_df["prediction"] == "H"
    ).sum()

    draw_predictions = (
        pred_df["prediction"] == "D"
    ).sum()

    away_predictions = (
        pred_df["prediction"] == "A"
    ).sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Matches",
            len(pred_df)
        )

    with c2:

        st.metric(
            "🏠 Home Wins",
            home_predictions
        )

    with c3:

        st.metric(
            "🤝 Draws",
            draw_predictions
        )

    with c4:

        st.metric(
            "✈️ Away Wins",
            away_predictions
        )

    st.divider()

    # ========================================================
    # MODEL DESCRIPTION
    # ========================================================

    if model_name == "XGBoost":

        st.info(
            "🚀 **XGBoost** — Gradient boosting model trained "
            "for 3-class match-result prediction."
        )

    elif model_name == "Random Forest":

        st.info(
            "🌲 **Random Forest** — Ensemble of decision trees "
            "used to classify match outcomes."
        )

    else:

        st.info(
            "📈 **Logistic Regression** — Multiclass linear "
            "classification model."
        )

    # ========================================================
    # MATCH CARDS
    # ========================================================

    st.subheader(
        f"⚽ {round_name} Predictions"
    )

    for i, match in pred_df.iterrows():

        prediction = match["prediction"]

        if prediction == "H":

            winner = match["home"]

        elif prediction == "A":

            winner = match["away"]

        else:

            winner = "Draw"

        with st.container(border=True):

            # -----------------------------------------------
            # HEADER
            # -----------------------------------------------

            header_col1, header_col2 = st.columns(
                [3, 1]
            )

            with header_col1:

                date_text = ""

                if pd.notna(match["date"]):

                    date_text = (
                        pd.to_datetime(
                            match["date"]
                        ).strftime("%d %b %Y")
                    )

                st.caption(
                    f"Match {i + 1} • {date_text}"
                )

                st.subheader(
                    f"{match['home']}  vs  {match['away']}"
                )

            with header_col2:

                if prediction == "H":

                    st.success(
                        f"🏆 {winner}"
                    )

                elif prediction == "A":

                    st.success(
                        f"🏆 {winner}"
                    )

                else:

                    st.warning(
                        "🤝 Draw"
                    )

            # -----------------------------------------------
            # PROBABILITIES
            # -----------------------------------------------

            st.markdown(
                "**Model probability**"
            )

            p1, p2, p3 = st.columns(3)

            with p1:

                st.metric(
                    f"🏠 {match['home']}",
                    f"{match['home_prob']:.1f}%"
                )

            with p2:

                st.metric(
                    "🤝 Draw",
                    f"{match['draw_prob']:.1f}%"
                )

            with p3:

                st.metric(
                    f"✈️ {match['away']}",
                    f"{match['away_prob']:.1f}%"
                )

            # Probability bar

            probability_df = pd.DataFrame(
                {
                    match["home"]: [
                        match["home_prob"]
                    ],

                    "Draw": [
                        match["draw_prob"]
                    ],

                    match["away"]: [
                        match["away_prob"]
                    ]
                }
            )

            st.bar_chart(
                probability_df,
                height=120
            )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.divider()

    download_df = pred_df.copy()

    download_df["Prediction"] = (
        download_df["prediction"]
        .map(RESULT_NAMES)
    )

    download_df = download_df.rename(
        columns={
            "home": "Home Team",
            "away": "Away Team",
            "home_prob": "Home Win %",
            "draw_prob": "Draw %",
            "away_prob": "Away Win %"
        }
    )

    csv = download_df.to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download Round Predictions",
        csv,
        file_name=
        f"{round_name.replace(' ', '_')}_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================
# SINGLE MATCH PREDICTOR
# ============================================================

elif page == "📊 Model Evaluation":

    st.header("📊 Model Evaluation")

    st.markdown(
        """
        Compare the performance of the three machine learning models
        used in the FIFA World Cup prediction system.
        """
    )

    # ============================================================
    # MODEL METRICS
    # ============================================================

    evaluation_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy": [
            0.608,
            0.596,
            0.599
        ],
        "Macro F1": [
            0.456,
            0.471,
            0.450
        ]
    })

    # ============================================================
    # TOP METRICS
    # ============================================================

    best_accuracy = evaluation_df.loc[
        evaluation_df["Accuracy"].idxmax()
    ]

    best_f1 = evaluation_df.loc[
        evaluation_df["Macro F1"].idxmax()
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🏆 Best Accuracy",
            f"{best_accuracy['Accuracy'] * 100:.1f}%",
            best_accuracy["Model"]
        )

    with col2:
        st.metric(
            "🎯 Best Macro F1",
            f"{best_f1['Macro F1']:.3f}",
            best_f1["Model"]
        )

    with col3:
        st.metric(
            "📊 Evaluation Samples",
            "3,598"
        )

    st.divider()

    # ============================================================
    # MODEL COMPARISON
    # ============================================================

    st.subheader("🤖 Model Comparison")

    comparison_display = evaluation_df.copy()

    comparison_display["Accuracy"] = (
        comparison_display["Accuracy"] * 100
    ).round(1)

    comparison_display["Macro F1"] = (
        comparison_display["Macro F1"]
    ).round(3)

    comparison_display = comparison_display.rename(
        columns={
            "Accuracy": "Accuracy (%)"
        }
    )

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True
    )

    # ============================================================
    # ACCURACY CHART
    # ============================================================

    st.subheader("📈 Accuracy Comparison")

    accuracy_chart = evaluation_df.set_index(
        "Model"
    )[["Accuracy"]]

    st.bar_chart(
        accuracy_chart
    )

    # ============================================================
    # MACRO F1 CHART
    # ============================================================

    st.subheader("🎯 Macro F1 Comparison")

    f1_chart = evaluation_df.set_index(
        "Model"
    )[["Macro F1"]]

    st.bar_chart(
        f1_chart
    )

    st.info(
        """
        **Why Macro F1?**

        Macro F1 calculates the F1 score independently for Away,
        Draw, and Home results and then gives each class equal
        importance. This is useful here because the dataset contains
        fewer Draw results than Home results.
        """
    )

    st.divider()

    # ============================================================
    # CLASSIFICATION REPORTS
    # ============================================================

    st.subheader(
        "📋 Classification Performance"
    )

    report_data = {

        "Logistic Regression": pd.DataFrame(
            {
                "Precision": [
                    0.578,
                    0.387,
                    0.626
                ],
                "Recall": [
                    0.647,
                    0.015,
                    0.872
                ],
                "F1 Score": [
                    0.610,
                    0.028,
                    0.729
                ]
            },
            index=[
                "Away",
                "Draw",
                "Home"
            ]
        ),

        "Random Forest": pd.DataFrame(
            {
                "Precision": [
                    0.570,
                    0.300,
                    0.631
                ],
                "Recall": [
                    0.615,
                    0.058,
                    0.846
                ],
                "F1 Score": [
                    0.592,
                    0.097,
                    0.723
                ]
            },
            index=[
                "Away",
                "Draw",
                "Home"
            ]
        ),

        "XGBoost": pd.DataFrame(
            {
                "Precision": [
                    0.568,
                    0.316,
                    0.621
                ],
                "Recall": [
                    0.643,
                    0.015,
                    0.856
                ],
                "F1 Score": [
                    0.603,
                    0.028,
                    0.720
                ]
            },
            index=[
                "Away",
                "Draw",
                "Home"
            ]
        )
    }

    selected_model = st.selectbox(
        "Select Model",
        list(report_data.keys())
    )

    selected_report = report_data[selected_model]

    st.dataframe(
        selected_report.style.format(
            "{:.3f}"
        ),
        use_container_width=True
    )

    # ============================================================
    # CLASS METRICS CHART
    # ============================================================

    st.subheader(
        f"📊 {selected_model} — Class Performance"
    )

    st.bar_chart(
        selected_report
    )

    st.divider()

    # ============================================================
    # CONFUSION MATRICES
    # ============================================================

    st.subheader("🔲 Confusion Matrix")

    confusion_matrices = {

        "Logistic Regression": np.array([
            [693, 11, 367],
            [297, 12, 518],
            [210, 8, 1482]
        ]),

        "Random Forest": np.array([
            [689, 6, 376],
            [301, 12, 514],
            [224, 20, 1456]
        ]),

        "XGBoost": np.array([
            [685, 0, 386],
            [291, 0, 536],
            [207, 0, 1493]
        ])
    }

    cm = confusion_matrices[selected_model]

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Away",
            "Actual Draw",
            "Actual Home"
        ],
        columns=[
            "Predicted Away",
            "Predicted Draw",
            "Predicted Home"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # ============================================================
    # HEATMAP
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    image = ax.imshow(cm)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))

    ax.set_xticklabels(
        ["Away", "Draw", "Home"]
    )

    ax.set_yticklabels(
        ["Away", "Draw", "Home"]
    )

    ax.set_xlabel(
        "Predicted Result"
    )

    ax.set_ylabel(
        "Actual Result"
    )

    ax.set_title(
        f"{selected_model} Confusion Matrix"
    )

    for i in range(3):

        for j in range(3):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    st.pyplot(
        fig
    )

    st.divider()

    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    st.subheader(
        "⚖️ Evaluation Dataset Distribution"
    )

    class_distribution = pd.DataFrame(
        {
            "Result": [
                "Away",
                "Draw",
                "Home"
            ],
            "Matches": [
                1071,
                827,
                1700
            ]
        }
    )

    st.bar_chart(
        class_distribution.set_index(
            "Result"
        )
    )

    st.warning(
        """
        **Key observation:** Draws are considerably harder for the
        models to identify than Home and Away results. The confusion
        matrices show that many actual Draws are classified as either
        Home or Away.
        """
    )

    # ============================================================
    # FINAL TAKEAWAY
    # ============================================================

    st.divider()

    st.subheader("💡 Key Findings")

    st.markdown(
        """
        **Logistic Regression**
        - Highest overall accuracy in this evaluation.
        - Strong performance on Home results.
        - Very weak at identifying Draws.

        **Random Forest**
        - Slightly lower accuracy than Logistic Regression.
        - Better Draw recall than the other two models.
        - Balanced ensemble approach.

        **XGBoost**
        - Strong Away and Home classification.
        - Very weak Draw detection.
        - Performs competitively with the other models.

        **Overall:** Accuracy alone does not tell the whole story.
        Macro F1 is useful because it evaluates all three result
        classes equally.
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FIFA World Cup Prediction Project • "
    "Machine Learning • Football Analytics"
)