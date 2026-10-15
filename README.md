# TCS Stock Price Prediction

## Project Overview
This project is a machine learning-based stock price analysis dashboard developed using Python and Streamlit. It analyzes historical Tata Consultancy Services (TCS) stock data and demonstrates next-observation closing-price estimates using regression models.

## Features
- Historical TCS closing-price visualization
- 20-day and 50-day moving-average analysis
- Interactive stock-price estimation form
- Random Forest model integration
- Comparison of Random Forest, Linear Regression, and previous-close baseline
- Historical dataset preview and CSV download

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
- yfinance

## Project Structure
- `app.py` — Streamlit dashboard
- `data/` — Historical stock dataset and data collection scripts
- `models/` — Saved trained model
- `notebook/` — Data analysis and model evaluation
- `src/` — Data collection source code
- `requirements.txt` — Python dependencies

## How to Run
1. Install Python.
2. Download or clone this repository.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run the dashboard using `streamlit run app.py`.

## Model Evaluation
The project compares regression models using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). In the current evaluation, the previous-close baseline performs slightly better than the tested machine-learning models.

## Limitations
The estimates are for educational demonstration only. They are not verified live-market forecasts or investment recommendations. Historical performance does not guarantee future results.
