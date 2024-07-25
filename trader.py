#Currently Works for NVDA, IBM, AAPL, URI, SIRI





import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import logging
import yfinance as yf
from buy import Buy  # Assuming Buy class is defined in buy.py

logging.basicConfig(level=logging.INFO)  # Example logging setup

# Global variables
sum = 20000
difList = []
buyList = []

def buyShare(company, allocatedMoney, dataset, threshold=0.02):
    global sum
    close_value = dataset['Close'].iloc[-1]
    amount = allocatedMoney // close_value
    actualMoney = amount * close_value
    if sum == 0 or actualMoney > sum:
        logging.info(f"Insufficient funds to buy {amount} shares of {company}.")
        return None
    expected_gain = (dataset['Close'].iloc[-1] - dataset['Close'].iloc[-2]) / dataset['Close'].iloc[-2]
    if expected_gain < threshold:
        logging.info(f"Expected gain {expected_gain:.4f} is below the threshold {threshold}. No purchase made.")
        return None
    sum -= actualMoney
    buyInstance = Buy(close_value, amount, company, dataset.index[-1])
    buyList.append(buyInstance)
    logging.info(f"Bought {amount} shares of {company} at {close_value:.2f}, expected gain {expected_gain:.4f}")
    return buyInstance
# def buyShare(company, allocatedMoney, dataset, threshold=0.02):
#     global sum
#     close_value = dataset['Close'].iloc[-1]
#     amount = min(allocatedMoney // close_value, sum // close_value)  # Adjust the number of shares based on available funds
#     actualMoney = amount * close_value
#     if amount == 0:
#         logging.info(f"Insufficient funds to buy any shares of {company}.")
#         return None
#     expected_gain = (dataset['Close'].iloc[-1] - dataset['Close'].iloc[-2]) / dataset['Close'].iloc[-2]
#     if expected_gain < threshold:
#         logging.info(f"Expected gain {expected_gain:.4f} is below the threshold {threshold}. No purchase made.")
#         return None
#     sum -= actualMoney
#     buyInstance = Buy(close_value, amount, company, dataset.index[-1])
#     buyList.append(buyInstance)
#     logging.info(f"Bought {amount} shares of {company} at {close_value:.2f}, expected gain {expected_gain:.4f}")
#     return buyInstance

def sell(buyInstance, percentage, stop_loss=0.05, take_profit=0.1):
    global sum
    current_price = dataset['Close'].iloc[-1]
    price_diff = (current_price - buyInstance.purchasePrice) / buyInstance.purchasePrice
    if price_diff <= -stop_loss:
        totalSell = buyInstance.quantity
    elif price_diff >= take_profit:
        totalSell = buyInstance.quantity
    else:
        totalSell = round(buyInstance.quantity * percentage)
    if totalSell > 0:
        logging.info(f"Selling {totalSell} shares of {buyInstance.company}")
        sum += totalSell * current_price
        buyInstance.quantity -= totalSell
        if buyInstance.quantity == 0:
            buyList.remove(buyInstance)
    else:
        logging.info(f"No shares sold for {buyInstance.company} as conditions not met.")

def loadDatasetAndModel(company):
    try:
        model_path = f'ExcelFiles/{company}/{company}.h5'
        model = load_model(model_path)

        # Fetch data from Yahoo Finance
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(months=3)  # Last 3 months

        start_data_whole = end_date - pd.DateOffset(years=10)
        datasetWhole = yf.download(company, start=start_data_whole, end=end_date)

        dataset = yf.download(company, start=start_date, end=end_date)
        
        logging.info(f"Data for {company} loaded successfully with shape: {dataset.shape}")
        print(dataset.index.min())  # Print the earliest date in the dataset
        print(dataset.index.max())  # Print the latest date in the dataset
        print(dataset.head())       # Print the first few rows of the dataset

        return model, dataset, datasetWhole
    
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
        logging.info(f"Future predictions: {future_predictions.flatten()}")
        return future_predictions.flatten()
    
    except ValueError as e:
        logging.error(f"Prediction error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

def weekSort(dataset):
    try:
        dataset.sort_index(inplace=True)
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(months=3)
        last_three_months = dataset.loc[(dataset.index >= start_date) & (dataset.index <= end_date)]
        
        logging.info(f"Filtered data from {start_date} to {end_date}: {last_three_months.head()}")
        
        weekly_data_fri = last_three_months.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        logging.info(f"Weekly data (Fridays) sorted: {weekly_data_fri}")
        return weekly_data_fri
    
    except Exception as e:
        logging.error(f"Week sorting error: {e}")

def traderBuy(dataset, company):
    try:
        futurePredictions = load_and_predict(model, dataset)
        if futurePredictions is None:
            logging.error(f"Failed to get future predictions for {company}")
            return

        weekDif = futurePredictions[4] - futurePredictions[0]
        difList.append(weekDif)
        
        if weekDif / dataset['Close'].iloc[-1] > 0.01:
            buyInstance = buyShare(company, 10000, dataset)
            if buyInstance:
                logging.info(f"Purchase: {buyInstance}")
                buyList.append(buyInstance)
        else:
            logging.info(f"Predicted to decrease or insufficient expected gain: {weekDif:.2f}")
    
    except Exception as e:
        logging.error(f"Error in traderBuy function: {e}")

def traderSell(dataset):
    try:
        closeToday = dataset['Close'].iloc[-1]
        for buy in buyList:
            if buy.date == dataset.index[-1]:
                continue
            else:
                dif = closeToday - buy.purchasePrice
                if dif > 5:
                    logging.info(f"Stock price increased since purchase. Selling stock at profit: {buy.purchasePrice}, {closeToday}, {dif}, {buy.quantity}")
                    sell(buy, .75)
                elif dif > 0 and dif <= 5:
                    logging.info(f"Stock price increased slightly. Selling half the stock: {buy.purchasePrice}, {closeToday}, {dif}, {buy.quantity}")
                    sell(buy, 0.5)
                else:
                    logging.info(f"No significant price increase for {buy.companyName}. Holding stock.")
    except Exception as e:
        logging.error(f"Error in traderSell function: {e}")

def buyListClean(buyList):
    return [buy for buy in buyList if buy.quantity > 0]

def tradeLoop(dataset, buyList, company, datasetWhole):
    try:
        # Sorting and filtering data
        datasetWeek = weekSort(dataset)
        fridays = datasetWeek.index[datasetWeek.index.weekday == 4]  # Friday only
        
        logging.info(f"Fridays in the dataset: {fridays}")
        
        # Ensure we have the full data available
        full_data = datasetWhole.copy()
        
        for friday in fridays:
            # Data up to the current Friday
            full_data_until_friday = full_data.loc[:friday]
            
            # Ensure at least 60 days of data are available
            if len(full_data_until_friday) < 60:
                logging.info(f"Not enough data to trade until {friday}.")
                continue

            # Clean up the buy list
            buyList = buyListClean(buyList)
            
            # Make predictions and trade decisions
            traderBuy(full_data_until_friday, company)
            
            # Optionally sell if conditions are met
            traderSell(full_data_until_friday)
        
        logging.info("Trade loop completed for company: {}".format(company))
    
    except Exception as e:
        logging.error(f"Error in tradeLoop function: {e}")

# Example usage
companyName = input("Enter company ticker symbol: ")
model, dataset, datasetWhole = loadDatasetAndModel(companyName)

if model is not None and not dataset.empty:
    tradeLoop(dataset, buyList, companyName, datasetWhole)
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
# for buy in buyList:
#     print("Purchase Price: ", buy.purchasePrice)
#     print("Quantity: ", buy.quantity)
#     print("Date: ", buy.date)
#     print("\n")


