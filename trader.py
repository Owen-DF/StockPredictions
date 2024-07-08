import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from buy import Buy

sum = 100000

def buy(company, allocatedMoney, dataset):
    global sum
    close_value = dataset['Close'].iloc[-1]
    amount = allocatedMoney // close_value
    actualMoney = amount * close_value
    sum -= actualMoney

    buyInstance = Buy(close_value, amount, company)

    sum -= amount
    print(buyInstance)
    return buyInstance



def sell(company, amount):
    global sum
    sum += amount
    print(f"Sold {amount} of {company} stock for ${amount}. Remaining balance: ${sum}")


#look at future 5 day sequence. If it is predicted to increase, hold current values, buy more
#if it is predicted to decrease, sell all stocks valued more than the final decrease
def loadDatasetAndModel(company):
    if company == "IBM":
        model = load_model('IBM.h5')
        dataset = pd.read_csv('ExcelFiles/IBM/IBM.csv', index_col='Date', parse_dates=['Date'])
    elif company =="United Rentals":
        model = load_model('stock_price_predictor_UnitedRentals.h5')
        dataset = pd.read_csv('ExcelFiles/UR/URI.csv', index_col='Date', parse_dates=['Date'])
    
    return model, dataset

model, dataset = loadDatasetAndModel("United Rentals")

def load_and_predict(model, dataset):



    # Prepare the data as before
    last_60_days = dataset["Close"][-60:].values.reshape(-1, 1)
    
    # Scale the data
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_last_60_days = sc.fit_transform(last_60_days)

    # Predict the future
    future_predictions = []
    current_input = scaled_last_60_days

    for _ in range(5):  # Predict next 5 days
        current_input = current_input.reshape((1, 60, 1))
        future_pred = model.predict(current_input)
        future_predictions.append(future_pred[0, 0])

        future_pred_reshaped = np.reshape(future_pred, (1, 1, 1))
        current_input = np.append(current_input[:, 1:, :], future_pred_reshaped, axis=1)

    future_predictions = sc.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    print("Future Prediction: ", future_predictions.flatten())

    # Plot the predictions if needed
    #plot_future_predictions(future_predictions)
    return future_predictions.flatten()

def plot_future_predictions(future_predictions):
    plt.plot(future_predictions, color='green', label='Predicted Future Stock Price')
    plt.title('Stock Price Future Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


difList = []
buyList = []

#MONDAY PROGRAM
futurePredictions = load_and_predict("United Rentals")

# If the stock price is predicted to increase, hold current values, buy more

weekDif = futurePredictions[4] - futurePredictions[0]
difList.append(weekDif)

if weekDif > 0:
    print("Predicted to increase")

    buyInstance = buy("United Rentals", 10000, dataset)
    print(buyInstance)
    buyList.append(buyInstance)



if weekDif < 0:
    print("Predicted to decrease")
    if difList[-1]<0:
        



    