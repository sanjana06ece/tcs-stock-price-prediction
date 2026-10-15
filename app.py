
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths

# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "tcs_stock_data.csv"
MODEL_PATH = BASE_DIR / "random_forest_stock_model.pkl"


st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 TCS Stock Price Prediction")
st.caption("Machine Learning Minor Project | Random Forest Regression")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load and clean dataset
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, skiprows=[1, 2])
    df.rename(columns={"Price": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = [
        "Adj Close", "Close", "High", "Low", "Open", "Volume"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Date", "Open", "High", "Low", "Close", "Volume"]
    )

    return df.sort_values("Date").reset_index(drop=True)

try:
    model = load_model()
    data = load_data()

    if data.empty:
        st.error("The dataset is empty. Please check the CSV file.")
        st.stop()

    latest = data.iloc[-1]

    # Predict using the latest available observation
    features = ["Open", "High", "Low", "Close", "Volume"]
    input_data = pd.DataFrame(
        [[latest[col] for col in features]],
        columns=features
    )

    prediction = float(model.predict(input_data)[0])

    # Dashboard metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Recorded Close",
        f"₹{latest['Close']:,.2f}"
    )

    col2.metric(
        "Model's Next-Observation Estimate",
        f"₹{prediction:,.2f}",
        f"₹{prediction - latest['Close']:,.2f}"
    )

    col3.metric("Historical Records", f"{len(data):,}")

    st.info(
        f"Latest date in dataset: {latest['Date'].date()}. "
        "The estimate is for the next observation after this date, "
        "not a verified live-market quote. It is educational only."
    )

    # Historical price chart
    st.subheader("Historical Closing Prices")

    chart_data = data.set_index("Date")["Close"]
    st.line_chart(chart_data)

    # Moving averages
    st.subheader("Moving Average Analysis")

    chart_df = data[["Date", "Close"]].copy()
    chart_df["20-Day Average"] = chart_df["Close"].rolling(20).mean()
    chart_df["50-Day Average"] = chart_df["Close"].rolling(50).mean()

    st.line_chart(
        chart_df.set_index("Date")[
            ["Close", "20-Day Average", "50-Day Average"]
        ]
    )

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(data.tail(10), use_container_width=True)

    st.download_button(
        "Download Cleaned Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_tcs_stock_data.csv",
        mime="text/csv"
    )

    st.caption(
        "Disclaimer: Predictions are estimates, not financial advice. "
        "Historical performance does not guarantee future results."
    )

except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")

except Exception as error:
    st.error(f"Something went wrong: {error}")

# Interactive stock price prediction
st.divider()
st.header("Interactive Stock Price Prediction")

st.write(
    "Enter historical market values to estimate the next "
    "observation's closing price using the trained model."
)

with st.form("stock_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        open_price = st.number_input(
            "Opening Price (₹)",
            min_value=0.01,
            value=float(latest["Open"]),
            step=1.0,
        )
        high_price = st.number_input(
            "Highest Price (₹)",
            min_value=0.01,
            value=float(latest["High"]),
            step=1.0,
        )
        low_price = st.number_input(
            "Lowest Price (₹)",
            min_value=0.01,
            value=float(latest["Low"]),
            step=1.0,
        )

    with col2:
        close_price = st.number_input(
            "Closing Price (₹)",
            min_value=0.01,
            value=float(latest["Close"]),
            step=1.0,
        )
        volume = st.number_input(
            "Trading Volume",
            min_value=0,
            value=int(latest["Volume"]),
            step=1000,
        )

    submitted = st.form_submit_button(
        "Predict Stock Price",
        type="primary",
    )

if submitted:
    if high_price < max(open_price, low_price, close_price):
        st.error(
            "The highest price must be at least as high "
            "as the opening, lowest, and closing prices."
        )
    elif low_price > min(open_price, high_price, close_price):
        st.error(
            "The lowest price must not exceed the opening, "
            "highest, or closing prices."
        )
    else:
        user_input = pd.DataFrame(
            [[open_price, high_price, low_price, close_price, volume]],
            columns=["Open", "High", "Low", "Close", "Volume"],
        )

        user_prediction = float(model.predict(user_input)[0])

        st.success(
            f"Estimated next-observation closing price: "
            f"₹{user_prediction:,.2f}"
        )
        st.caption(
            "Educational model estimate only. This is not a "
            "live-market forecast or investment recommendation."
        )

# Model performance comparison
st.divider()
st.header("Model Performance Evaluation")

st.write(
    "The models were evaluated on a chronological test set. "
    "Lower MAE and RMSE indicate smaller prediction errors."
)

evaluation = pd.DataFrame({
    "Model": [
        "Original Random Forest",
        "Improved Random Forest",
        "Linear Regression",
        "Previous-close baseline"
    ],
    "MAE (₹)": [38.75, 43.95, 32.71, 32.23],
    "RMSE (₹)": [52.38, 58.50, 46.24, 45.92]
})

st.dataframe(evaluation, hide_index=True, use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 4))

evaluation.plot(
    x="Model",
    y=["MAE (₹)", "RMSE (₹)"],
    kind="bar",
    ax=ax
)

ax.set_title("Model Error Comparison")
ax.set_ylabel("Error (₹)")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.info(
    "In this evaluation, the previous-close baseline performed best "
    "on both MAE and RMSE. The Random Forest models did not outperform "
    "this baseline. Further tuning and validation are needed."
)
