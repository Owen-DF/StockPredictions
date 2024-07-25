# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD, Adam
import math
from sklearn.metrics import mean_squared_error
import yfinance as yf
import datetime as dt

# Get today's date
today = dt.date.today()
print(f"Today's date: {today}")

# Calculate the date 10 years ago from today
start = today - pd.DateOffset(years=10)


company = input("Enter the company name (IBM, URI, NVDA, AAPL, SIRI, BA): ")
if company == "IBM":
    data = yf.download('IBM', start=start, end=today)
elif company == "URI":
    data = yf.download('URI', start=start, end=today)
elif company == "NVDA":
    data = yf.download('NVDA', start=start, end=today)
elif company == "AAPL":
    data = yf.download('AAPL', start=start, end=today)
elif company == "SIRI":
    data = yf.download('SIRI', start=start, end=today)
elif company == "BA":
    data = yf.download('BA', start=start, end=today)
# Calculate the end date that is 3 months before the maximum date in the dataset
datasetEnd = data.index.max() - pd.DateOffset(months=3)
print(f"End date excluding the last 3 months: {datasetEnd}")

# Filter the dataset to exclude the last 3 months
dataset_filtered = data[data.index <= datasetEnd]

# Calculate the quantile date to split the dataset
quantileDate = dataset_filtered.index.to_series().quantile(0.8)

# Split the dataset into training and testing sets without overlapping
training_set = dataset_filtered.loc[:quantileDate, 'Open'].values
test_set = dataset_filtered.loc[quantileDate + pd.Timedelta(days=1):, 'Open'].values

# Scale the training set
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set.reshape(-1, 1))

# Prepare the training data
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the LSTM model
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))
optimizer = SGD(learning_rate=.01)

# Compiling the RNN
regressor.compile(optimizer=optimizer, loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train, y_train, epochs=50, batch_size=32)

# Saving the model
if company == "IBM":
    regressor.save('ExcelFiles/IBM/IBM.h5')
elif company == "URI":
    regressor.save('ExcelFiles/URI/URI.h5')
elif company == "NVDA":
    regressor.save('ExcelFiles/NVDA/NVDA.h5')
elif company == "AAPL":
    regressor.save('ExcelFiles/AAPL/AAPL.h5')
elif company == "SIRI":
    regressor.save('ExcelFiles/SIRI/SIRI.h5')
elif company == "BA":
    regressor.save('ExcelFiles/BA/BA.h5')

# Prepare the inputs for prediction on the test set
dataset_total = pd.concat((dataset_filtered["Open"][:quantileDate], dataset_filtered["Open"][quantileDate + pd.Timedelta(days=1):]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# Predicting the stock prices for the test set
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.figure(figsize=(14, 5))
plt.plot(dataset_filtered.loc[quantileDate + pd.Timedelta(days=1):].index, test_set, color='red', label='Real '+company + 'Stock Price')
plt.plot(dataset_filtered.loc[quantileDate + pd.Timedelta(days=1):].index, predicted_stock_price, color='blue', label='Predicted '+company+' Stock Price')
plt.title( str(company + ' Stock Price Prediction'))
plt.xlabel('Time')
plt.ylabel(company + ' Stock Price')
plt.legend()
plt.show()

# Evaluating the model
rmse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
print(f'Root Mean Squared Error: {rmse}')


