import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import math
from sklearn.metrics import mean_squared_error

def readFile(company):
    print(company)
    match company:
        case "IBM":
            dataset = pd.read_csv('ExcelFiles/IBM/IBM.csv', index_col='Date', parse_dates=['Date'])
            prepareData(dataset, company)
        case "United Rentals":
            dataset = pd.read_csv('ExcelFiles/UR/URI.csv', index_col='Date', parse_dates=['Date'])
            prepareData(dataset, company)
        case "Nvidia":
            dataset = pd.read_csv('ExcelFiles/NVIDIA/NVDA.csv', index_col='Date', parse_dates=['Date'])
            prepareData(dataset, company)
        case _:
            print("boop")

def prepareData(training_set, test_set, company, dataset):


    # Scale the data
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set.reshape(-1, 1))

    X_train = []
    y_train = []
    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train
    # lstmModel(X_train, y_train, dataset, test_set, sc, company)

# Example call to the prepareData function
# companyName = "YourCompany"
# dataset = load your dataset here
# prepareData(dataset, companyName)


def lstmModel(X_train, y_train, company):
    # The LSTM architecture
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

    # Compiling the RNN
    regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
    # Fitting to the training set
    regressor.fit(X_train, y_train, epochs=50, batch_size=32)


    if company == "IBM":
        regressor.save('ExcelFiles/IBM/IBM.h5')
    elif company == "United Rentals":
        regressor.save('ExcelFiles/UR/URI.h5')
    elif company == "Nvidia":
        regressor.save('ExcelFiles/NVIDIA/Nvidia.h5')

    x_test(dataset, test_set, regressor)

def x_test(dataset, test_set, regressor):
    sc = MinMaxScaler(feature_range=(0, 1))
    dataset_total = pd.concat((dataset["Close"][:'2023-06-30'], dataset["Close"]['2023-07-01':]), axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    # Preparing X_test and predicting the prices
    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    plot_predictions(test_set, predicted_stock_price)

    return_rmse(test_set, predicted_stock_price)

    # Predicting the next 5 days
    predict_future(dataset, sc, regressor)

def plot_predictions(test, predicted):
    plt.plot(test, color='red', label='Real IBM Stock Price')
    plt.plot(predicted, color='blue', label='Predicted IBM Stock Price')
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.show()

def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    
def predict_future(dataset, sc, regressor):
    future_days = 5
    last_60_days = dataset["Close"][-60:].values.reshape(-1, 1)
    scaled_last_60_days = sc.transform(last_60_days)

    future_predictions = []
    current_input = scaled_last_60_days

    for _ in range(future_days):
        current_input = current_input.reshape((1, 60, 1))
        future_pred = regressor.predict(current_input)
        future_predictions.append(future_pred[0, 0])

        future_pred_reshaped = np.reshape(future_pred, (1, 1, 1))
        current_input = np.append(current_input[:, 1:, :], future_pred_reshaped, axis=1)

    future_predictions = sc.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    print("Future Prediction: ", future_predictions.flatten())
    # plot_future_predictions(future_predictions)

def plot_future_predictions(future_predictions):
    plt.plot(future_predictions, color='green', label='Predicted Future IBM Stock Price')
    plt.title('IBM Stock Price Future Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.show()

# Example usage
# readFile('IBM')
