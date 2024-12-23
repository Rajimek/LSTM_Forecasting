# PROJECT_LSTM

## Overview
This project focuses on using machine learning techniques, specifically Long Short-Term Memory (LSTM) models, to predict the daily settle price of the S&P 500 index (SPX). The project incorporates historical price data and technical indicators to build a robust predictive model. Additionally, it explores statistical methods (e.g., ARIMA) as benchmarks for comparison.

## Project Structure

### Data
The `data` directory contains all the input data files required for model training and evaluation:
- **bloomberg/**: Includes raw and pre-processed CSV files:
  - `historical_prices_spx.csv`: Historical daily prices for the SPX index.
  - `index_metrics_CONCCONF.csv`: Consumer confidence index data.
  - `index_metrics_NAPMPMI.csv`: Manufacturing PMI data.
  - `index_metrics_SPX.csv`: SPX-specific technical metrics (e.g., P/E ratio, earnings yield).
  - `index_metrics_USGG10YR.csv`: U.S. 10-year treasury bond yield data.
  - `index_metrics_VIX.csv`: Volatility Index (VIX) data.
  - `forward_filled_merged_data.csv`: Pre-processed and merged dataset for modeling.

### Source Code
The `src` directory contains Python scripts for data processing and modeling:
- **`data.py`**: Handles data loading, cleaning, and pre-processing tasks.
- **`lstm_favorable_features.py`**: Identifies and selects favorable features for the LSTM model.
- **`LSTM_SW.py`**: Implementation of the LSTM model for time series forecasting.
- **`model_arima.py`**: Includes the ARIMA model as a benchmark for the LSTM predictions.


## Requirements
- Python 3.11.8
- Poetry 1.8.5

## Installation
To set up the project environment, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd PROJECT_LSTM
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Activate the Virtual Environment**:
   ```bash
   poetry shell
   ```

4. **Run the Project**:
   ```bash
   poetry run python src/main.py
   ```
