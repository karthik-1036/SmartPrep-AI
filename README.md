# SmartPrep AI

A beginner-friendly Streamlit app for cloud kitchen sales analysis and demand forecasting.

## What It Does
- Upload an Excel file of sales
- Show sales by item and peak hours
- Predict tomorrow's demand per item using a simple ML model
- Provide a plain-English explanation

## Setup
1. Create a virtual environment (optional but recommended)
2. Install dependencies:

```bash
pip install streamlit pandas scikit-learn openpyxl
```

## Run
```bash
streamlit run app.py
```

## Expected Columns (Flexible)
The app lets you map columns manually. Common column names:
- Item: `item`, `menu`, `product`, `dish`
- Quantity: `qty`, `quantity`, `count`, `units`
- Date: `date`
- Time: `time`
- Datetime: `datetime`, `timestamp`, `created`

## Notes
- The model learns from patterns like day-of-week and recent sales.
- If there isn’t enough history, it falls back to a 7-day average.
