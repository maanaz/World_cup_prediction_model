import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FIFA World Cup AI Predictor",
    page_icon="⚽",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

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

RESULT_NAMES = {
    "A": "Away Win",
    "D": "Draw",
    "H": "Home Win"
}

ROUND_FILES = {
    "Group Stage": "wc_group_stage_features.csv",
    "Round of 32": "r32_features.csv",
    "Round of 16": "r16_features.csv",
    "Quarter Finals": "qf_features.csv",
    "Semi Finals": "semi_features.csv",
    "Final": "final_features.csv"
}

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(filename):
    """Load and clean a CSV file."""
    csv_path = BASE_DIR / filename

    df = pd.read_csv(csv_path)

    # Remove accidental pandas index columns.
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


def load_round(round_name):
    """Load the feature dataset for a tournament round."""
    if round_name not in ROUND_FILES:
        raise ValueError(f"Unknown round: {round_name}")

    return load_csv(ROUND_FILES[round_name])


# ============================================================
# FASTAPI FUNCTIONS
# ============================================================

def test_api_connection():
    """Check whether the FastAPI backend is running."""
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )
        return response
    except requests.RequestException:
        return None


def predict_match(round_name, match_index, model_name):
    """Send a classification prediction request to FastAPI."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "round": round_name,
                "match_index": match_index,
                "model": model_name
            },
            timeout=30
        )

        return response

    except requests.RequestException as error:
        st.error(f"Could not connect to FastAPI: {error}")
        return None


def predict_score(round_name, match_index):
    """Send a Poisson score prediction request to FastAPI."""
    try:
        response = requests.post(
            f"{API_URL}/predict/score",
            json={
                "round": round_name,
                "match_index": match_index
            },
            timeout=30
        )

        return response

    except requests.RequestException as error:
        st.error(f"Could not connect to FastAPI: {error}")
        return None


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_classification_result(data, home_team, away_team):
    """Display the classification result returned by FastAPI."""

    prediction = data["prediction"]

    probabilities = data["probabilities"]

    st.success(
        f"🏆 Prediction: **{RESULT_NAMES[prediction]}**"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            home_team,
            f"{probabilities['home'] * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Draw",
            f"{probabilities['draw'] * 100:.1f}%"
        )

    with col3:
        st.metric(
            away_team,
            f"{probabilities['away'] * 100:.1f}%"
        )

    chart_df = pd.DataFrame(
        {
            "Outcome": [
                home_team,
                "Draw",
                away_team
            ],
            "Probability": [
                probabilities["home"],
                probabilities["draw"],
                probabilities["away"]
            ]
        }
    )

    st.bar_chart(
        chart_df.set_index("Outcome")
    )


def display_score_result(data, home_team, away_team):
    """Display the Poisson score prediction returned by FastAPI."""

    expected_goals = data["expected_goals"]

    home_lambda = expected_goals["home"]
    away_lambda = expected_goals["away"]

    score_df = pd.DataFrame(
        data["top_scores"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            f"{home_team} Expected Goals",
            f"{home_lambda:.2f}"
        )

    with col2:
        st.metric(
            f"{away_team} Expected Goals",
            f"{away_lambda:.2f}"
        )

    best_score = data["most_likely_score"]

    st.success(
        f"Most likely score: "
        f"**{best_score['home_goals']} - "
        f"{best_score['away_goals']}** "
        f"({best_score['probability'] * 100:.1f}%)"
    )

    top_scores = score_df.copy()

    top_scores["Score"] = (
        top_scores["home_goals"]
        .astype(int)
        .astype(str)
        + " - "
        + top_scores["away_goals"]
        .astype(int)
        .astype(str)
    )

    top_scores["Probability"] = (
        top_scores["probability"] * 100
    ).round(2)

    st.dataframe(
        top_scores[
            ["Score", "Probability"]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Prediction Settings")

if st.sidebar.button("🔌 Test FastAPI"):
    api_response = test_api_connection()

    if api_response is not None and api_response.status_code == 200:
        st.sidebar.success(
            f"FastAPI: {api_response.json()['status']}"
        )
    elif api_response is not None:
        st.sidebar.error(
            f"FastAPI error: {api_response.status_code}"
        )
    else:
        st.sidebar.error(
            "FastAPI is not reachable."
        )

page = st.sidebar.radio(
    "Go to",
    [
        "🏆 Tournament Predictions",
        "⚽ Match Predictor",
        "📊 Model Evaluation"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title("⚽ FIFA World Cup 2026")
st.subheader(
    "Machine Learning Match Prediction & Football Analytics"
)


# ============================================================
# PAGE 1 — TOURNAMENT PREDICTIONS
# ============================================================

if page == "🏆 Tournament Predictions":

    st.header("🏆 Tournament Center")

    st.markdown(
        "Explore how the machine learning models predict "
        "every stage of the 2026 FIFA World Cup."
    )

    st.divider()

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL DESCRIPTION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREDICT ALL MATCHES THROUGH FASTAPI
    # --------------------------------------------------------

    predictions = []

    progress = st.progress(0)

    for position, (_, row) in enumerate(df.iterrows()):

        response = predict_match(
            round_name,
            position,
            model_name
        )

        if response is None:
            st.stop()

        if response.status_code != 200:
            st.error(
                f"FastAPI error for match {position}: "
                f"{response.json()}"
            )
            st.stop()

        data = response.json()

        probabilities = data["probabilities"]

        predictions.append(
            {
                "date": row["date"],
                "home": row["home_team"],
                "away": row["away_team"],
                "prediction": data["prediction"],
                "home_prob": probabilities["home"] * 100,
                "draw_prob": probabilities["draw"] * 100,
                "away_prob": probabilities["away"] * 100
            }
        )

        progress.progress(
            (position + 1) / len(df)
        )

    progress.empty()

    pred_df = pd.DataFrame(predictions)

    # --------------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MATCH CARDS
    # --------------------------------------------------------

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

            header_col1, header_col2 = st.columns(
                [3, 1]
            )

            with header_col1:

                date_text = ""

                if pd.notna(match["date"]):
                    date_text = pd.to_datetime(
                        match["date"]
                    ).strftime("%d %b %Y")

                st.caption(
                    f"Match {i + 1} • {date_text}"
                )

                st.subheader(
                    f"{match['home']}  vs  {match['away']}"
                )

            with header_col2:

                if prediction == "D":
                    st.warning("🤝 Draw")
                else:
                    st.success(
                        f"🏆 {winner}"
                    )

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

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

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
        file_name=(
            f"{round_name.replace(' ', '_')}"
            "_predictions.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# PAGE 2 — SINGLE MATCH PREDICTOR
# ============================================================

elif page == "⚽ Match Predictor":

    st.header("⚽ Single Match Predictor")

    # --------------------------------------------------------
    # MATCH SELECTION
    # --------------------------------------------------------

    round_name = st.selectbox(
        "Tournament Round",
        list(ROUND_FILES.keys())
    )

    df = load_round(round_name)

    match_labels = []

    for i, row in df.iterrows():
        match_labels.append(
            f"{i}: {row['home_team']} vs {row['away_team']}"
        )

    selected_match = st.selectbox(
        "Select Match",
        match_labels
    )

    selected_index = int(
        selected_match.split(":")[0]
    )

    row = df.iloc[selected_index]

    st.divider()

    st.subheader(
        f"{row['home_team']} vs {row['away_team']}"
    )

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    model_name = st.selectbox(
        "Prediction Model",
        [
            "XGBoost",
            "Random Forest",
            "Logistic Regression"
        ]
    )

    predict_button = st.button(
        "🔮 Predict",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------

    if predict_button:

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        st.subheader("🏆 Match Result Prediction")

        classification_response = predict_match(
            round_name,
            selected_index,
            model_name
        )

        if classification_response is None:
            st.stop()

        if classification_response.status_code != 200:
            st.error(
                "FastAPI classification error: "
                f"{classification_response.json()}"
            )
            st.stop()

        classification_data = (
            classification_response.json()
        )

        display_classification_result(
            classification_data,
            row["home_team"],
            row["away_team"]
        )

        # ====================================================
        # POISSON SCORE PREDICTION
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Poisson Score Prediction"
        )

        score_response = predict_score(
            round_name,
            selected_index
        )

        if score_response is None:
            st.stop()

        if score_response.status_code != 200:
            st.error(
                "FastAPI score prediction error: "
                f"{score_response.json()}"
            )
            st.stop()

        score_data = score_response.json()

        display_score_result(
            score_data,
            row["home_team"],
            row["away_team"]
        )


# ============================================================
# PAGE 3 — MODEL EVALUATION
# ============================================================

elif page == "📊 Model Evaluation":

    st.header("📊 Model Evaluation")

    st.markdown(
        """
        Compare the performance of the three machine learning
        models used in the FIFA World Cup prediction system.
        """
    )

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    evaluation_df = pd.DataFrame(
        {
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
        }
    )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ACCURACY CHART
    # --------------------------------------------------------

    st.subheader("📈 Accuracy Comparison")

    accuracy_chart = evaluation_df.set_index(
        "Model"
    )[["Accuracy"]]

    st.bar_chart(
        accuracy_chart
    )

    # --------------------------------------------------------
    # MACRO F1 CHART
    # --------------------------------------------------------

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

        Macro F1 calculates the F1 score independently for
        Away, Draw, and Home results and then gives each class
        equal importance. This is useful here because the
        dataset contains fewer Draw results than Home results.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CLASSIFICATION REPORTS
    # --------------------------------------------------------

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

    selected_report = report_data[
        selected_model
    ]

    st.dataframe(
        selected_report.style.format(
            "{:.3f}"
        ),
        use_container_width=True
    )

    # --------------------------------------------------------
    # CLASS METRICS CHART
    # --------------------------------------------------------

    st.subheader(
        f"📊 {selected_model} — Class Performance"
    )

    st.bar_chart(
        selected_report
    )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    st.subheader("🔲 Confusion Matrix")

    confusion_matrices = {

        "Logistic Regression": np.array(
            [
                [693, 11, 367],
                [297, 12, 518],
                [210, 8, 1482]
            ]
        ),

        "Random Forest": np.array(
            [
                [689, 6, 376],
                [301, 12, 514],
                [224, 20, 1456]
            ]
        ),

        "XGBoost": np.array(
            [
                [685, 0, 386],
                [291, 0, 536],
                [207, 0, 1493]
            ]
        )
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

    # --------------------------------------------------------
    # CONFUSION MATRIX HEATMAP
    # --------------------------------------------------------

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

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

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
        **Key observation:** Draws are considerably harder
        for the models to identify than Home and Away results.
        The confusion matrices show that many actual Draws
        are classified as either Home or Away.
        """
    )

    # --------------------------------------------------------
    # FINAL TAKEAWAY
    # --------------------------------------------------------

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
