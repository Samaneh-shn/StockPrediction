# StockPredictor

## Stock Market Prediction Analysis

This project involves predicting stock price movements using advanced machine learning models. By applying Random Forest and Long Short-Term Memory (LSTM) networks, the project aims to forecast significant price movements and predict future stock prices with enhanced accuracy.

### Table of Contents

1. [Overview](#overview)
2. [Data Collection](#data-collection)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Training and Evaluation](#model-training-and-evaluation)
   - [Random Forest Model](#random-forest-model)
   - [LSTM Network for Price Prediction](#lstm-network-for-price-prediction)
6. [Results](#results)
7. [Visualizations](#visualizations)
8. [How to Use](#how-to-use)
9. [Conclusion](#conclusion)
10. [Future Work](#future-work)

## Overview

The StockPredictor project utilizes machine learning techniques to analyze and predict stock market movements. It focuses on forecasting significant price changes and daily closing prices using a combination of Random Forest classifiers and LSTM neural networks. These models are trained on historical data from major stock tickers, providing insights into future market trends.

## Data Collection

- Data is collected using the Yahoo Finance API.
- The dataset includes historical stock prices from January 2021 to the present.
- Data fields include daily opening, closing, high, low prices, and trading volume for each stock.

## Data Preprocessing

### Percentage Change Calculation

- Daily percentage changes in the adjusted close prices are calculated to capture day-to-day volatility.
- A significant daily percentage change is defined as a movement greater than 5%. This threshold helps in identifying high volatility days that are critical for predicting future price movements.

### Lag Features

- Lag features are created by shifting the significant movement indicator for five days.
- These features help in capturing temporal dependencies, essential for time-series forecasting in financial markets.

## Feature Engineering

- Lagged features of significant movements are used as predictors.
- This approach is based on the hypothesis that past volatility patterns provide insights into future price behaviors, a common assumption in financial time series analysis.

## Model Training and Evaluation

### Random Forest Model

- A Random Forest Classifier is trained to predict whether there will be a significant price movement (greater than 5%) on a given day.
- This model is effective in handling non-linear relationships and is robust against overfitting in high-dimensional data.
- **Evaluation Metrics:** The model's performance is assessed using the F1 score, focusing on the balance between precision and recall, which is particularly useful for imbalanced datasets where significant movements are rare.

### LSTM Network for Price Prediction

- An LSTM (Long Short-Term Memory) network is utilized to predict future stock prices based on historical data.
- **Model Setup:** The LSTM network is designed to capture sequential data patterns, making it well-suited for stock price predictions.
- **Data Scaling:** Input features are normalized using MinMaxScaler to ensure that the LSTM model receives scaled data, optimizing weight updates during training.
- **Training:** The model is trained to predict the next day’s closing price based on patterns observed over the past 100 days. The training process minimizes prediction errors, measured by the Mean Squared Error (MSE) loss function.

## Results

- The Random Forest model achieved an F1 score of **72%**, indicating a good balance between precision and recall in predicting significant price movements.
- The LSTM network achieved a **Mean Squared Error (MSE) of 0.004** on the test dataset, demonstrating effective short-term prediction capabilities.

## Visualizations

- Model performance and significant price movement patterns are visualized using graphs and plots.
- These visualizations help in better understanding the model's predictions and the stock market's volatility.

## How to Use

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/StockPredictor.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the data collection script to fetch historical stock data using the Yahoo Finance API.
4. Execute the preprocessing scripts to prepare the data for modeling.
5. Train the Random Forest model and LSTM network using the respective training scripts.
6. Evaluate the models and visualize the results using the evaluation scripts.

## Conclusion

The StockPredictor project demonstrates the application of machine learning techniques to forecast stock market movements. By utilizing Random Forest and LSTM models, the project successfully predicts significant price movements and daily closing prices, aiding in investment decision-making.

## Future Work

- Extend the model to include additional features such as macroeconomic indicators, news sentiment analysis, and technical indicators.
- Experiment with other machine learning models such as GRU (Gated Recurrent Unit) and CNN (Convolutional Neural Networks) for potentially improved accuracy.
- Implement real-time prediction capabilities using streaming data for practical investment applications.

