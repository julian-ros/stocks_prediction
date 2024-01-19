# %%
# Imports necessary libraries and downloads the historical data for BTC-USD from Yahoo Finance, then displays the first 10 rows of the data.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
data = yf.download(tickers = 'BTC-USD', start = '2012-01-01',end = '2023-12-31')
data.head(10)

# %%
# Displays the last 10 rows of the data.
data.tail(10)

# %%
# Adds various technical indicators to the data (RSI, EMAF, EMAM, EMAS), calculates the target variable 
# (the difference between the adjusted close and open prices), creates a binary target class variable, and drops unnecessary columns.
data['RSI']=ta.rsi(data.Close, length=15)
data['EMAF']=ta.ema(data.Close, length=20)
data['EMAM']=ta.ema(data.Close, length=100)
data['EMAS']=ta.ema(data.Close, length=150)

data['Target'] = data['Adj Close']-data.Open
data['Target'] = data['Target'].shift(-1)

data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

data['TargetNextClose'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace = True)
# Create a 'dates' array from the original DataFrame before dropping it
original_dates = pd.to_datetime(data['Date'])

data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

# %%
#  Creates a new dataset with the first 11 columns of the data and displays the first 20 rows.
data_set = data.iloc[:, 0:11]#.values
pd.set_option('display.max_columns', None)
data_set.head(20)

# %%
# Scales the dataset to a range of 0 to 1 using MinMaxScaler.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
sc_adj_close = MinMaxScaler(feature_range=(0,1))
sc_adj_close.fit(data[['Adj Close']])
data_set_scaled = sc.fit_transform(data_set)
print(data_set_scaled)

# %%
# Creates a 3D array X to hold the past backcandles days of data for each of the 8 features, and a 1D array y for the target variable.
X = []
backcandles = 30
print(data_set_scaled.shape[0])
for j in range(8):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i-backcandles:i, j])

X=np.moveaxis(X, [0], [2])
X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-1])
y=np.reshape(yi,(len(yi),1))
print(X)
print(X.shape)
print(y)
print(y.shape)

# %%
# split data into train test sets
splitlimit = int(len(X)*0.8)
print(splitlimit)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train)

# %%
# Defines and compiles a LSTM model, then fits the model to the training data.
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
#tf.random.set_seed(20)
np.random.seed(10)

lstm_input = Input(shape=(backcandles, 8), name='lstm_input')
inputs = LSTM(50, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)

# %%
# This cell uses the trained model to make predictions on the test set (X_test).
# Also runs a whole prediction for the entire dataset
# It then prints the first 10 pairs of predicted and actual values to provide a quick check on the model’s performance.
dates = data.index[splitlimit + backcandles:]
y_pred = model.predict(X_test)
y_pred_all = model.predict(X)
for i in range(10):
    print(y_pred[i], y_test[i])

# %%
# Plots the prediction

# Reset the index of the 'dates' array
#dates.reset_index(drop=True, inplace=True)

# Then split it into training and testing sets just like you did with 'X' and 'y'
dates_train, dates_test = original_dates[:splitlimit], original_dates[splitlimit+backcandles:]

# Normalize the data for BTC Price from the MixMaxScaler logic

y_test_adj_close = y_test[:, 0:1]  # Select the first feature
y_pred_adj_close = y_pred[:, 0:1]  # Select the first feature


dates_test = original_dates[splitlimit+backcandles:]
y_test_inv = sc_adj_close.inverse_transform(y_test_adj_close)
y_pred_inv = sc_adj_close.inverse_transform(y_pred_adj_close)

print(f'dates shape: {original_dates.shape}')
print(f'dates_test shape: {dates_test.shape}')
print(f'y_test_inv shape: {y_test_inv.shape}')

plt.figure(figsize=(16,8))
plt.plot(dates_test, y_test_inv, color = 'black', label = 'Test')
plt.plot(dates_test, y_pred_inv, color = 'green', label = 'pred')
plt.legend()
plt.show()


# %%
# Predict the whole timeseries
dates_all = original_dates[backcandles:]
y_pred_all = model.predict(X)

# Normalize the data for BTC Price from the MixMaxScaler logic

y_test_all_adj_close = y[:, 0:1]  # Select the first feature
y_pred__all_adj_close = y[:, 0:1]  # Select the first feature

y_test_all_inv = sc_adj_close.inverse_transform(y_test_all_adj_close)
y_pred_all_inv = sc_adj_close.inverse_transform(y_pred__all_adj_close)


plt.figure(figsize=(16,8))
plt.plot(dates_all, y_test_all_inv, color = 'black', label = 'Actual')
plt.plot(dates_all, y_pred_all_inv, color = 'green', label = 'Predicted')
plt.legend()
plt.show()

# %%
# Predict the whole timeseries
dates_all = original_dates[backcandles:]
y_pred_all = model.predict(X)

# Normalize the data for BTC Price from the MixMaxScaler logic
y_test_all_adj_close = y[:, 0:1]  # Select the first feature
y_pred__all_adj_close = y[:, 0:1]  # Select the first feature

y_test_all_inv = sc_adj_close.inverse_transform(y_test_all_adj_close)
y_pred_all_inv = sc_adj_close.inverse_transform(y_pred__all_adj_close)

# Flatten y_pred_all_inv to 1D array
y_pred_all_inv_flattened = y_pred_all_inv.flatten()

# Prediction for 2024
forecast_days = 366 # for 1 year (including leap year)
input_data = X[-1,:,:]
forecasted_data = []

# Multi-step forecasting
for i in range(forecast_days):
    predicted = model.predict(input_data.reshape(1, backcandles, 8))
    forecasted_data.append(predicted[0])
    input_data = np.roll(input_data, -1)
    input_data[-1] = predicted

# Convert to numpy array and inverse transform
forecasted_data = np.array(forecasted_data)
forecasted_data_inv = sc_adj_close.inverse_transform(forecasted_data)

# Reshape to 1D array
forecasted_data_inv = forecasted_data_inv.reshape(-1)

# Append the forecasted data for 2024 to y_pred_all_inv
y_pred_all_inv_extended = np.concatenate((y_pred_all_inv_flattened, forecasted_data_inv))

# Generate future dates for 2024 and append to dates_all
future_dates = pd.Series(pd.date_range(start='2024-01-01', end='2024-12-31'))
dates_all_extended = pd.concat([dates_all, future_dates])

# Plot the data
plt.figure(figsize=(16,8))
plt.plot(dates_all_extended, y_pred_all_inv_extended, color = 'green', label = 'Predicted')
plt.plot(dates_all, y_test_all_inv, color = 'black', label = 'Actual')
plt.legend()
plt.show()



