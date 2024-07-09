import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from buy import Buy  # Assuming Buy class is defined in buy.py
import math
import logging

logging.basicConfig(level=logging.INFO)  # Example logging setup

# Global variables (consider encapsulating in a class for better organization)
sum = 100000
difList = []
buyList = []

def buyShare(company, allocatedMoney, dataset):
    global sum
    close_value = dataset['Close'].iloc[-1]
    amount = allocatedMoney // close_value
    actualMoney = amount * close_value
    sum -= actualMoney

    buyInstance = Buy(close_value, amount, company, dataset.index[-1])
    sum -= amount
    return buyInstance

def sell(buyInstance, percentage):
    global sum
    totalSell = round(buyInstance.quantity * percentage)
    print(f"Selling {totalSell} shares")
    sum += totalSell * buyInstance.purchasePrice
    buyInstance.quantity -= totalSell

def loadDatasetAndModel(company):
    try:
        if company == "IBM":
            model_path = 'ExcelFiles/IBM/IBM.h5'
            dataset_path = 'ExcelFiles/IBM/IBM.csv'
        elif company == "United Rentals":
            model_path = 'ExcelFiles/UR/URI.h5'
            dataset_path = 'ExcelFiles/UR/URI.csv'
        elif company == "Nvidia":
            model_path = 'ExcelFiles/NVIDIA/Nvidia.h5'
            dataset_path = 'ExcelFiles/NVIDIA/NVDA.csv'
        else:
            raise ValueError("Company not supported.")
        
        model = load_model(model_path)
        dataset = pd.read_csv(dataset_path, index_col='Date', parse_dates=['Date'])
        
        return model, dataset
    
    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
    except ValueError as e:
        logging.error(f"Invalid company or dataset: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

def load_and_predict(model, dataset):
    try:
        if len(dataset) < 60:
            raise ValueError("Not enough data to make predictions. Need at least 60 days of data.")
        
        last_60_days = dataset["Close"][-60:].values.reshape(-1, 1)
        sc = MinMaxScaler(feature_range=(0, 1))
        scaled_last_60_days = sc.fit_transform(last_60_days)
        
        future_predictions = []
        current_input = scaled_last_60_days
        
        for _ in range(5):
            current_input = current_input.reshape((1, 60, 1))
            future_pred = model.predict(current_input)
            future_predictions.append(future_pred[0, 0])
            
            future_pred_reshaped = np.reshape(future_pred, (1, 1, 1))
            current_input = np.append(current_input[:, 1:, :], future_pred_reshaped, axis=1)
        
        future_predictions = sc.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        # logging.info(f"Future Prediction: {future_predictions.flatten()}")
        return future_predictions.flatten()
    
    except ValueError as e:
        logging.error(f"Prediction error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

def weekSort(dataset):
    try:
        dataset.sort_index(inplace=True)
        last_three_months = dataset.loc[dataset.index >= (dataset.index.max() - pd.DateOffset(months=3))]
        
        weekly_data_fri = last_three_months.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        return weekly_data_fri
    
    except Exception as e:
        logging.error(f"Week sorting error: {e}")

def traderBuy(dataset):
    try:
        futurePredictions = load_and_predict(model, dataset)
        weekDif = futurePredictions[4] - futurePredictions[0]
        difList.append(weekDif)
        
        if weekDif > 0:
            buyInstance = buyShare("United Rentals", 10000, dataset)
            # logging.info(f"Predicted to increase: {weekDif}, Buy instance: {buyInstance}")
            logging.info(f"Purchase:  {buyInstance}")
            buyList.append(buyInstance)
        elif weekDif < 0:
            logging.info(f"Predicted to decrease: {weekDif}")
    
    except Exception as e:
        logging.error(f"Error in traderBuy function: {e}")

def traderSell(dataset):
    try:
        closeToday = dataset['Close'].iloc[-1]
        for buy in buyList:
            if buy.date == dataset.index[-1]:
                continue
            else:
                print(buy)
                dif = closeToday - buy.purchasePrice
                if dif > 5:
                    logging.info(f"Stock price increased since purchase. Selling stock at profit: {buy.purchasePrice}, {closeToday}, {dif}, {buy.quantity}")
                    sell(buy, .75)
                elif dif > 0 and dif <= 5:
                    logging.info(f"Stock price increased slightly. Selling half the stock: {buy.purchasePrice}, {closeToday}, {dif}, {buy.quantity}")
                    sell(buy, 0.5)
                else:
                    # logging.info(f"Stock price stayed the same or decreased. Holding stock: {buy.purchasePrice}, {closeToday}, {dif}, {buy.quantity}")
                    pass
    except Exception as e:
        logging.error(f"Error in traderSell function: {e}")


def buyListClean(buyList):
    for buy in buyList:
        if buy.quantity == 0:
            buyList.remove(buy)
    return buyList


def tradeLoop(dataset, buyList):
    try:
        datasetWeek = weekSort(dataset)
        fridays = datasetWeek[datasetWeek.index.weekday == 4].index  # Fridays in the dataset
        for friday in fridays:
            full_data_until_friday = dataset.loc[:friday]  # Data up to the current Friday
            if len(full_data_until_friday) < 60:
                continue  # Skip if there are not enough data points
            buyList = buyListClean(buyList)
            traderBuy(full_data_until_friday)
            traderSell(full_data_until_friday)
    
    except Exception as e:
        logging.error(f"Error in tradeLoop function: {e}")

# Example usage
companyName = input("Enter company name: ")
model, dataset = loadDatasetAndModel(companyName)

if model is not None and not dataset.empty:
    tradeLoop(dataset, buyList)
else:
    print("Error loading model or dataset. Exiting...")
# Calculate total assets and sum after trading
currentAsset = 0
buyList = buyListClean(buyList)
for buy in buyList:
    currentAsset += buy.quantity * dataset['Close'].iloc[-1]


total = sum + currentAsset
logging.info(f"Current asset: {currentAsset}")
logging.info(f"Sum: {sum}")
logging.info(f"Total: {total}")


print("\nSummary: Current Price", dataset['Close'].iloc[-1])
for buy in buyList:
    print("Purchase Price: ", buy.purchasePrice)
    print("Quantity: ", buy.quantity)
    print("Date: ", buy.date)
    print("\n")

