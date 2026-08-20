from gdown.download import download

import os
import textwrap

import pandas as pd
import requests
import streamlit as st
import tempfile

# ╔════════════════════════════════════════════════════════════╗
# ║ ⚙️ CONFIG
# ╚════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Sentiment Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "https://sentiment-analysis-baseline-44767566354.europe-west1.run.app"
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", DEFAULT_API_URL))
REVIEWS_DATA_URL = os.getenv(
    "REVIEWS_DATA_URL",
    "https://drive.google.com/file/d/1m9nG79MoPaIf6ubuin8rX3Y9-YsUpjhx/view?usp=share_link",
)

SENTIMENT_META = {
    "Negative": {
        "emoji": "😫",
        "css_class": "negative",
        "subtitle": "The model detected a negative sentiment.",
    },
    "Neutral": {
        "emoji": "😐",
        "css_class": "neutral",
        "subtitle": "The model detected a neutral sentiment.",
    },
    "Positive": {
        "emoji": "😁",
        "css_class": "positive",
        "subtitle": "The model detected a positive sentiment.",
    },
}


# ╔════════════════════════════════════════════════════════════╗
# ║ 🎨 THEME
# ╚════════════════════════════════════════════════════════════╝

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(121, 82, 255, 0.14), transparent 28%),
                radial-gradient(circle at 90% 20%, rgba(0, 190, 255, 0.10), transparent 24%),
                #0d111b;
            color: #f5f7ff;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1322 0%, #0b1020 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 24px;
            background: linear-gradient(
                135deg,
                rgba(111, 76, 255, 0.18),
                rgba(18, 20, 31, 0.92)
            );
            border: 1px solid rgba(255, 255, 255, 0.09);
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.4rem;
        }

        .hero-kicker {
            font-size: 0.82rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #a89cff;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }

        .hero-subtitle {
            max-width: 760px;
            color: #d2d8e6;
            margin-top: 1rem;
            font-size: 1.05rem;
            line-height: 1.65;
        }

        .hero-subtitle {
            max-width: 760px;
            color: #b8bccb;
            margin-top: 1rem;
            font-size: 1.05rem;
            line-height: 1.65;
        }

        .soft-card {
            background: rgba(24, 28, 44, 0.96);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 20px;
            padding: 1.25rem 1.35rem;
            height: 100%;
            color: #f3f6ff;
        }

        .result-card {
            border-radius: 24px;
            padding: 2rem;
            margin-top: 1rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(23, 27, 43, 0.98);
        }

        .result-emoji {
            font-size: 4.4rem;
            line-height: 1.1;
            margin-bottom: 0.4rem;
        }

        .result-label {
            font-size: 2.25rem;
            font-weight: 850;
            letter-spacing: -0.04em;
        }

        .result-subtitle {
            color: #aeb2c0;
            margin-top: 0.4rem;
        }

        .negative .result-label {
            color: #ff6b7a;
        }

        .neutral .result-label {
            color: #f5b942;
        }

        .positive .result-label {
            color: #48d597;
        }

        .confidence {
            margin: 1.3rem auto 0 auto;
            max-width: 520px;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            font-weight: 700;
            font-size: 1.05rem;
        }

        .confidence-low {
            color: #ff6674;
            background: rgba(255, 71, 87, 0.10);
            border: 1px solid rgba(255, 71, 87, 0.25);
        }

        .confidence-medium {
            color: #ffb347;
            background: rgba(255, 159, 67, 0.10);
            border: 1px solid rgba(255, 159, 67, 0.25);
        }

        .confidence-high {
            color: #50dc99;
            background: rgba(46, 213, 115, 0.10);
            border: 1px solid rgba(46, 213, 115, 0.25);
        }

        .metric-label {
            color: #c3cada;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        div[data-testid="stMetric"] {
            background: rgba(24, 28, 44, 0.94);
            border: 1px solid rgba(255,255,255,0.10);
            padding: 1rem;
            border-radius: 18px;
        }

        div[data-testid="stMetricLabel"] p {
            color: #eef3ff !important;
            font-weight: 750 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #dfe7ff !important;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        button[data-baseweb="tab"] {
            border-radius: 14px;
            padding-left: 1.15rem;
            padding-right: 1.15rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            min-height: 3rem;
            font-weight: 750;
            border: 1px solid rgba(139, 166, 255, 0.42);
        }

        .stButton > button[kind="primary"] {
            background: #7c5cff;
            color: #ffffff;
            border-color: #9f8cff;
        }

        .stButton > button[kind="primary"]:hover {
            background: #8e73ff;
            border-color: #c2b7ff;
            color: #ffffff;
        }

        .stButton > button:disabled,
        .stButton > button[disabled] {
            background: #27314d !important;
            color: #c9d4ef !important;
            border: 1px solid rgba(139, 166, 255, 0.24) !important;
            opacity: 1 !important;
        }

        .small-note,
        .stCaption,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {
            color: #c7cfdd !important;
        }

        .class-chip {
            display: inline-block;
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            font-size: 0.90rem;
            font-weight: 700;
            color: #f5f7ff;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.14);
        }
        .chip-negative {
            background: rgba(255, 96, 109, 0.16);
            color: #ff9aa5;
            border: 1px solid rgba(255, 96, 109, 0.30);
        }

        .chip-neutral {
            background: rgba(255, 184, 77, 0.16);
            color: #ffd089;
            border: 1px solid rgba(255, 184, 77, 0.30);
        }

        .chip-positive {
            background: rgba(72, 213, 151, 0.16);
            color: #8ef0c0;
            border: 1px solid rgba(72, 213, 151, 0.30);
        }

        label, .stTextArea label, .stSelectbox label {
            color: #e9edf8 !important;
            font-weight: 600;
        }

        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div {
            background: #151a2a !important;
            color: #f5f7ff !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }

        .stTextArea textarea::placeholder {
            color: #98a3bd !important;
            opacity: 1 !important;
        }

        .stTextArea textarea:focus {
            border-color: rgba(159, 140, 255, 0.9) !important;
            box-shadow: 0 0 0 1px rgba(159, 140, 255, 0.55) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ╔════════════════════════════════════════════════════════════╗
# ║ 🧰 HELPERS
# ╚════════════════════════════════════════════════════════════╝


@st.cache_data(show_spinner=False)
def load_dataset(url: str) -> pd.DataFrame:

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    ) as tmp:
        temp_path = tmp.name

    download(
        url=url,
        output=temp_path,
        quiet=True,
    )

    df = pd.read_csv(temp_path)

    required_columns = {"Score", "Text"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The downloaded file is not the expected Amazon Reviews dataset."
        )

    if "Score" in df.columns and "Feeling" not in df.columns:
        df["Feeling"] = df["Score"].map(
            {
                1: 0,
                2: 0,
                3: 1,
                4: 2,
                5: 2,
            }
        )

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(
            df["Time"],
            unit="s",
            errors="coerce",
        )

    return df


def api_health() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.ok
    except requests.RequestException:
        return False


def predict_sentiment(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def confidence_ui(score: float) -> tuple[str, str, str]:
    if score < 0.50:
        return (
            "confidence-low",
            "😵",
            "Low confidence",
        )

    if score < 0.70:
        return (
            "confidence-medium",
            "🤨",
            "Moderate confidence",
        )

    return (
        "confidence-high",
        "✅",
        "High confidence",
    )


def sentiment_name(value: int) -> str:
    mapping = {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
    }
    return mapping.get(value, "Unknown")


# ╔════════════════════════════════════════════════════════════╗
# ║ 🧭 SIDEBAR
# ╚════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## 🧠 Sentiment Lab")
    st.caption("Amazon Reviews • NLP Demo")

    is_api_online = api_health()

    if is_api_online:
        st.success("API connected")
    else:
        st.error("API offline")

    st.markdown("---")

    st.markdown("**Model**")
    st.caption("CNN — Tokenization")

    st.markdown("**Classes**")
    st.markdown(
        """
        <span class="class-chip">😫 Negative</span>
        <span class="class-chip">😐 Neutral</span>
        <span class="class-chip">😁 Positive</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "Portfolio demonstration • Predictions are produced by a trained "
        "sentiment classification model."
    )


# ╔════════════════════════════════════════════════════════════╗
# ║ 🪧 HEADER
# ╚════════════════════════════════════════════════════════════╝

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">NLP • Deep Learning • FastAPI</div>
        <h1 class="hero-title">Sentiment Lab</h1>
        <div class="hero-subtitle">
            Explore an Amazon review sentiment classifier trained to distinguish
            <strong>negative</strong>, <strong>neutral</strong> and
            <strong>positive</strong> opinions.
            Test the model yourself or inspect the dataset behind the experiment.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ╔════════════════════════════════════════════════════════════╗
# ║ 📑 TABS
# ╚════════════════════════════════════════════════════════════╝

prediction_tab, dataset_tab = st.tabs(
    [
        "🔮 Prediction",
        "📊 Dataset",
    ]
)


# ╔════════════════════════════════════════════════════════════╗
# ║ 🔮 TAB — PREDICTION
# ╚════════════════════════════════════════════════════════════╝

with prediction_tab:
    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        st.markdown("### Try the classifier")
        st.caption(
            "Enter an English review or sentence. The API will return the predicted sentiment."
        )

        example = st.selectbox(
            "Need an example?",
            [
                "Write my own review",
                "I absolutely love this product. It works perfectly!",
                "The product is okay, nothing special but nothing terrible either.",
                "I am extremely disappointed. This was a complete waste of money.",
            ],
        )

        default_text = "" if example == "Write my own review" else example

        user_text = st.text_area(
            "Review",
            value=default_text,
            height=180,
            placeholder="Example: The product arrived quickly and the quality is amazing...",
            max_chars=2000,
        )

        chars = len(user_text)
        st.caption(f"{chars}/2000 characters")
        st.warning(
            "⚠️ Model limitation: the CNN may struggle with negation handling "
            "(e.g. 'not good', 'not happy'). In manual tests, predictions appear "
            "more stable on French reviews than short English sentences. "
            "Cross-language robustness remains an improvement area."
        )

        classify = st.button(
            "Analyze sentiment ✨",
            type="primary",
            disabled=not user_text.strip(),
        )

    with right:
        st.markdown("### How it works")
        st.markdown(
            """
            <div class="soft-card">
                <div class="metric-label">Inference pipeline</div><br>
                📝 Raw text<br><br>
                ↓<br><br>
                🧹 Cleaning<br><br>
                ↓<br><br>
                🔢 Tokenization + padding<br><br>
                ↓<br><br>
                🧠 CNN classifier<br><br>
                ↓<br><br>
                🎭 Sentiment + confidence
            </div>
            """,
            unsafe_allow_html=True,
        )

    if classify:
        if not is_api_online:
            st.error(
                f"The FastAPI backend is not reachable at `{API_URL}`. "
                "Start the API before running a prediction."
            )
        else:
            try:
                with st.spinner(
                    "Reading the emotional subtext... because apparently text has moods too."
                ):
                    result = predict_sentiment(user_text.strip())

                label = str(result.get("label", "Unknown")).title()
                confidence = float(
                    result.get(
                        "confiance",
                        result.get("probability", 0.0),
                    )
                )

                meta = SENTIMENT_META.get(
                    label,
                    {
                        "emoji": "🧠",
                        "css_class": "",
                        "subtitle": "Prediction returned by the model.",
                    },
                )

                confidence_class, confidence_emoji, confidence_label = confidence_ui(
                    confidence
                )

                result_html = textwrap.dedent(
                    f"""
                    <div class="result-card {meta["css_class"]}">
                        <div class="result-emoji">{meta["emoji"]}</div>
                        <div class="result-label">{label}</div>
                        <div class="result-subtitle">{meta["subtitle"]}</div>
                        <div class="confidence {confidence_class}">
                            {confidence_emoji}
                            {confidence_label}
                            &nbsp;•&nbsp;
                            {confidence:.1%}
                        </div>
                    </div>
                    """
                ).strip()
                st.markdown(result_html, unsafe_allow_html=True)

                st.progress(min(max(confidence, 0.0), 1.0))

                if confidence < 0.50:
                    st.caption(
                        "The model is uncertain about this prediction. "
                        "Interpret the result cautiously."
                    )
                elif confidence < 0.70:
                    st.caption("The model has moderate confidence in this prediction.")
                else:
                    st.caption(
                        "The model has relatively high confidence in this prediction."
                    )

            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response else None
                if status_code == 503:
                    st.error(
                        "The API is online, but prediction is temporarily unavailable. "
                        "The Cloud Run service likely needs more memory for TensorFlow."
                    )
                else:
                    st.error("The API returned an error while classifying the text.")
                with st.expander("Technical details"):
                    details = str(exc)
                    if exc.response is not None:
                        details += f"\n\nResponse body:\n{exc.response.text}"
                    st.code(details)

            except requests.RequestException as exc:
                st.error("Unable to contact the prediction API.")
                with st.expander("Technical details"):
                    st.code(str(exc))

            except (TypeError, ValueError, KeyError) as exc:
                st.error("The API response could not be interpreted.")
                with st.expander("Technical details"):
                    st.code(str(exc))


# ╔════════════════════════════════════════════════════════════╗
# ║ 📊 TAB — DATASET
# ╚════════════════════════════════════════════════════════════╝

with dataset_tab:
    st.markdown("### Amazon Reviews — Dataset Overview")
    st.caption(
        "A compact version of the exploratory analysis used before model training."
    )

    try:
        with st.spinner("Loading Amazon Reviews dataset..."):
            df = load_dataset(REVIEWS_DATA_URL)

        total_rows = len(df)
        total_cols = df.shape[1]
        missing_values = int(df.isna().sum().sum())
        duplicated_rows = int(df.duplicated().sum())

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Reviews", f"{total_rows:,}")
        m2.metric("Columns", total_cols)
        m3.metric("Missing values", f"{missing_values:,}")
        m4.metric("Duplicate rows", f"{duplicated_rows:,}")

        st.markdown("#### Preview")

        preview_cols = [
            col
            for col in ["ProductId", "Score", "Summary", "Text", "Feeling", "Time"]
            if col in df.columns
        ]

        st.dataframe(
            df[preview_cols].head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Class distribution")

        chart_left, chart_right = st.columns(2, gap="large")

        with chart_left:
            if "Score" in df.columns:
                st.markdown("**Amazon score distribution**")
                score_counts = df["Score"].value_counts().sort_index().rename("Reviews")
                st.bar_chart(score_counts)

        with chart_right:
            if "Feeling" in df.columns:
                st.markdown("**Sentiment distribution**")

                feeling_counts = (
                    df["Feeling"]
                    .value_counts()
                    .sort_index()
                    .rename(
                        index={
                            0: "Negative",
                            1: "Neutral",
                            2: "Positive",
                        }
                    )
                    .rename("Reviews")
                )

                st.bar_chart(feeling_counts)

        if "Feeling" in df.columns:
            proportions = df["Feeling"].value_counts(normalize=True).sort_index()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "😫 Negative",
                f"{proportions.get(0, 0):.1%}",
            )
            c2.metric(
                "😐 Neutral",
                f"{proportions.get(1, 0):.1%}",
            )
            c3.metric(
                "😁 Positive",
                f"{proportions.get(2, 0):.1%}",
            )

            st.caption(
                "The original dataset is strongly imbalanced, which is why "
                "class balancing is part of the modeling workflow."
            )

        if "Time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Time"]):
            st.markdown("#### Reviews over time")

            yearly_counts = (
                df["Time"].dt.year.value_counts().sort_index().rename("Reviews")
            )

            st.line_chart(yearly_counts)

        with st.expander("🔍 Dataset inspection"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Data types**")
                dtype_df = df.dtypes.astype(str).rename("dtype").to_frame()
                st.dataframe(dtype_df, use_container_width=True)

            with col_b:
                st.markdown("**Unique values**")
                unique_df = (
                    df.nunique()
                    .sort_values(ascending=False)
                    .rename("n_unique")
                    .to_frame()
                )
                st.dataframe(unique_df, use_container_width=True)

            st.markdown("**Missing values by column**")
            missing_df = pd.DataFrame(
                {
                    "missing": df.isna().sum(),
                    "missing_%": (df.isna().mean() * 100).round(2),
                }
            ).sort_values("missing", ascending=False)

            st.dataframe(
                missing_df,
                use_container_width=True,
            )

        with st.expander("📈 Descriptive statistics"):
            st.dataframe(
                df.describe(include="all").T,
                use_container_width=True,
            )

        with st.expander("🧪 Random review examples"):
            sample_size = min(5, len(df))

            random_rows = df.sample(
                sample_size,
                random_state=42,
            )

            display_cols = [
                col
                for col in ["Score", "Summary", "Text", "Feeling"]
                if col in random_rows.columns
            ]

            for _, row in random_rows[display_cols].iterrows():
                label = (
                    sentiment_name(int(row["Feeling"]))
                    if "Feeling" in row.index and pd.notna(row["Feeling"])
                    else "Unknown"
                )

                emoji = SENTIMENT_META.get(
                    label,
                    {"emoji": "📝"},
                )["emoji"]

                title = row.get("Summary", "Review")

                st.markdown(f"**{emoji} {label} — {title}**")
                st.write(row.get("Text", ""))
                st.divider()

    except Exception as exc:
        st.error("Unable to load the Amazon Reviews dataset.")

        with st.expander("Technical details"):
            st.code(str(exc))
# ╔════════════════════════════════════════════════════════════╗
# ║ 🧾 FOOTER
# ╚════════════════════════════════════════════════════════════╝

st.markdown("---")
st.caption("Sentiment Lab • CNN Tokenization • FastAPI • Streamlit • Portfolio Demo")
