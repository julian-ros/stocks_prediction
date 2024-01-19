# Bitcoin Price Prediction Model

This machine learning model predicts the future price of Bitcoin using historical data.

## Functionality

The model works by:
1. Downloading historical data for BTC-USD from Yahoo Finance.
2. Adding various technical indicators to the data (RSI, EMAF, EMAM, EMAS).
3. Calculating the target variable (the difference between the adjusted close and open prices).
4. Creating a binary target class variable.
5. Dropping unnecessary columns.
6. Scaling the dataset to a range of 0 to 1 using MinMaxScaler.
7. Creating a 3D array X to hold the past backcandles days of data for each of the 8 features, and a 1D array y for the target variable.
8. Splitting the data into train and test sets.
9. Defining and compiling a LSTM model, then fitting the model to the training data.
10. Using the trained model to make predictions on the test set (X_test).

## Requirements

The model requires the following Python libraries:
- numpy
- matplotlib
- pandas
- yfinance
- pandas_ta
- sklearn
- keras
- tensorflow

## How to Run

1. Ensure all required libraries are installed.
2. Run your junyper environment or installed required junyper plugins for vscode
3. Run the notebook cells. The model will automatically download the necessary data and begin training.

## Use Cases

This model can be used to predict the future price of Bitcoin, which can be useful for trading strategies and financial analysis. It can also be developed further for other goals such as:
- Intraday trading providing ticker data
- Multiple assets, crypto or regular market stocks

## Note

Please note that this model is for educational purposes only and should not be used as financial advice. Always do your own research before making any investment decisions.
