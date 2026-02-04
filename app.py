import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(page_title="SmartPrep AI - Demand Forecast", layout="wide")
st.title("SmartPrep AI - Cloud Kitchen Demand Forecast")
st.caption("Upload sales data, explore trends, and get a simple prediction for tomorrow.")


def guess_column(columns, keywords):
    for key in keywords:
        for col in columns:
            if key in col.lower():
                return col
    return None


def to_datetime(date_series, time_series=None):
    if time_series is None:
        return pd.to_datetime(date_series, errors="coerce")
    return pd.to_datetime(
        date_series.astype(str) + " " + time_series.astype(str), errors="coerce"
    )


uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if not uploaded:
    st.info("Please upload an Excel file to begin.")
    st.stop()

data = pd.read_excel(uploaded)
st.subheader("Preview")
st.dataframe(data.head(20), use_container_width=True)

cols = list(data.columns)
st.subheader("Select Columns")

default_item = guess_column(cols, ["item", "menu", "product", "dish"])
default_qty = guess_column(cols, ["qty", "quantity", "count", "units"])
default_sales = guess_column(cols, ["sales", "revenue", "amount", "total"])
default_date = guess_column(cols, ["date"])
default_time = guess_column(cols, ["time", "hour"])
default_datetime = guess_column(cols, ["datetime", "timestamp", "created"])

with st.expander("Column Mapping", expanded=True):
    item_col = st.selectbox("Item column", cols, index=cols.index(default_item) if default_item in cols else 0)
    qty_col = st.selectbox("Quantity column", cols, index=cols.index(default_qty) if default_qty in cols else 0)
    sales_col = st.selectbox(
        "Sales/Revenue column (optional)",
        ["(none)"] + cols,
        index=(["(none)"] + cols).index(default_sales) if default_sales in cols else 0,
    )
    datetime_col = st.selectbox(
        "Datetime column (optional)",
        ["(none)"] + cols,
        index=(["(none)"] + cols).index(default_datetime) if default_datetime in cols else 0,
    )
    date_col = st.selectbox(
        "Date column (optional if datetime is provided)",
        ["(none)"] + cols,
        index=(["(none)"] + cols).index(default_date) if default_date in cols else 0,
    )
    time_col = st.selectbox(
        "Time column (optional if datetime is provided)",
        ["(none)"] + cols,
        index=(["(none)"] + cols).index(default_time) if default_time in cols else 0,
    )


df = data.copy()
df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

if sales_col != "(none)":
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)

if datetime_col != "(none)":
    df["__datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
else:
    if date_col == "(none)":
        df["__datetime"] = pd.NaT
    else:
        time_series = None if time_col == "(none)" else df[time_col]
        df["__datetime"] = to_datetime(df[date_col], time_series)

df = df.dropna(subset=[item_col])

if df["__datetime"].notna().any():
    df["__date"] = df["__datetime"].dt.date
    df["__hour"] = df["__datetime"].dt.hour
else:
    df["__date"] = pd.NaT
    df["__hour"] = pd.NaT


st.subheader("Sales by Item")
item_sales = (
    df.groupby(item_col)[qty_col]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={qty_col: "Quantity"})
)
st.bar_chart(item_sales, x=item_col, y="Quantity", use_container_width=True)

st.subheader("Peak Hours")
if df["__hour"].notna().any():
    hourly = (
        df.groupby("__hour")[qty_col]
        .sum()
        .reset_index()
        .rename(columns={"__hour": "Hour", qty_col: "Quantity"})
        .sort_values("Hour")
    )
    st.line_chart(hourly, x="Hour", y="Quantity", use_container_width=True)
else:
    st.info("No time or datetime column found to compute peak hours.")


st.subheader("Predict Tomorrow's Demand")

if df["__date"].notna().any():
    daily = (
        df.groupby([item_col, "__date"])[qty_col]
        .sum()
        .reset_index()
        .rename(columns={qty_col: "DailyQty"})
    )

    # ML forecasting per item using time features + recent lags
    forecasts = []
    for item, group in daily.groupby(item_col):
        group = group.sort_values("__date").copy()
        raw_group = group.copy()
        group["date_dt"] = pd.to_datetime(group["__date"])
        group["dow"] = group["date_dt"].dt.dayofweek
        group["dom"] = group["date_dt"].dt.day
        group["month"] = group["date_dt"].dt.month
        group["trend"] = range(len(group))
        group["lag_1"] = group["DailyQty"].shift(1)
        group["lag_7"] = group["DailyQty"].shift(7)
        group["roll_7"] = group["DailyQty"].rolling(7).mean()
        group["roll_14"] = group["DailyQty"].rolling(14).mean()

        group = group.dropna()
        if len(group) < 12:
            recent = raw_group["DailyQty"].tail(7)
            prediction = recent.mean() if not recent.empty else 0
            forecasts.append({"Item": item, "PredictedQty": round(float(prediction), 2)})
            continue

        feature_cols = ["dow", "dom", "month", "trend", "lag_1", "lag_7", "roll_7", "roll_14"]
        X = group[feature_cols]
        y = group["DailyQty"]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        )
        model.fit(X, y)

        last_row = group.iloc[-1]
        last_date = last_row["date_dt"]
        tomorrow = last_date + pd.Timedelta(days=1)

        X_tomorrow = pd.DataFrame(
            [{
                "dow": tomorrow.dayofweek,
                "dom": tomorrow.day,
                "month": tomorrow.month,
                "trend": int(last_row["trend"]) + 1,
                "lag_1": last_row["DailyQty"],
                "lag_7": group.iloc[-7]["DailyQty"] if len(group) >= 7 else last_row["DailyQty"],
                "roll_7": group["DailyQty"].tail(7).mean(),
                "roll_14": group["DailyQty"].tail(14).mean(),
            }]
        )

        pred = model.predict(X_tomorrow)[0]
        forecasts.append({"Item": item, "PredictedQty": round(float(pred), 2)})

    forecast_df = pd.DataFrame(forecasts).sort_values("PredictedQty", ascending=False)
    st.dataframe(forecast_df, use_container_width=True)

    st.caption("Prediction method: Random Forest per item with time and lag features. Falls back to 7-day average if data is limited.")
else:
    st.info("No date or datetime column found to predict tomorrow. Add a date column to enable forecasting.")


st.subheader("AI Explanation")
if df["__date"].notna().any():
    explanation = (
        "I trained a small model per item using patterns like weekday, recent sales, and "
        "rolling averages. That model predicts tomorrow’s demand. If an item has too little "
        "history, I fall back to a simple 7-day average for stability."
    )
else:
    explanation = (
        "I couldn't generate a forecast because a date or datetime column is missing. "
        "If you provide dates, I can average the last 7 days to estimate tomorrow's demand."
    )

st.write(explanation)
