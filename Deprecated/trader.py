import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from buy import Buy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
sum = 20000
difList = []
buyList = []
transaction_log = []

def buyShare(company, allocatedMoney, dataset, threshold=0.02):
    global sum
    close_value = dataset['Close'].iloc[-1]
    amount = allocatedMoney // close_value
    actualMoney = amount * close_value
    logger.info(f"Attempting to buy {amount} shares of {company} at {close_value:.2f} for {actualMoney:.2f}")
    
    if sum == 0 or actualMoney > sum:
        logger.info(f"Not enough funds to buy {amount} shares. Available funds: {sum:.2f}")
        return None
    expected_gain = (dataset['Close'].iloc[-1] - dataset['Close'].iloc[-2]) / dataset['Close'].iloc[-2]
    if expected_gain < threshold:
        logger.info(f"Expected gain {expected_gain:.2f} is below the threshold {threshold:.2f}. Not buying.")
        return None
    sum -= actualMoney
    buyInstance = Buy(close_value, amount, company, dataset.index[-1])
    buyList.append(buyInstance)
    transaction_log.append(f"Bought {amount} shares of {company} at {close_value:.2f} for {actualMoney:.2f} on {dataset.index[-1]}")
    logger.info(f"Successfully bought {amount} shares of {company}")
    return buyInstance

def sell(buyInstance, percentage, stop_loss=0.05, take_profit=0.1):
    global sum
    current_price = dataset['Close'].iloc[-1]
    price_diff = (current_price - buyInstance.purchasePrice) / buyInstance.purchasePrice
    logger.info(f"Evaluating sell for {buyInstance.company} at {current_price:.2f}. Price difference: {price_diff:.2f}")
    
    if price_diff <= -stop_loss or price_diff >= take_profit:
        totalSell = buyInstance.quantity
    else:
        totalSell = round(buyInstance.quantity * percentage)
    if totalSell > 0:
        sum += totalSell * current_price
        transaction_log.append(f"Sold {totalSell} shares of {buyInstance.company} at {current_price:.2f} for {totalSell * current_price:.2f}")
        buyInstance.quantity -= totalSell
        logger.info(f"Sold {totalSell} shares of {buyInstance.company}. Remaining quantity: {buyInstance.quantity}")
        if buyInstance.quantity == 0:
            buyList.remove(buyInstance)

def loadDatasetAndModel(company):
    model_path = f'ExcelFiles/{company}/{company}.h5'
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)

    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=3)
    start_data_whole = end_date - pd.DateOffset(years=10)
    datasetWhole = yf.download(company, start=start_data_whole, end=end_date)
    dataset = yf.download(company, start=start_date, end=end_date)
    
    logger.info(f"Loaded dataset for {company} from {start_data_whole.date()} to {end_date.date()}")
    return model, dataset, datasetWhole

def load_and_predict(model, dataset):
    if len(dataset) < 60:
        logger.warning("Insufficient data length for prediction")
        return None
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
    logger.info(f"Predicted future prices: {future_predictions.flatten()}")
    return future_predictions.flatten()

def weekSort(dataset):
    dataset.sort_index(inplace=True)
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=3)
    last_three_months = dataset.loc[(dataset.index >= start_date) & (dataset.index <= end_date)]
    
    weekly_data_fri = last_three_months.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return weekly_data_fri

def traderBuy(dataset, company):
    futurePredictions = load_and_predict(model, dataset)
    if futurePredictions is None:
        return

    weekDif = futurePredictions[4] - futurePredictions[0]
    difList.append(weekDif)
    
    if weekDif / dataset['Close'].iloc[-1] > 0.01:
        buyInstance = buyShare(company, 10000, dataset)
        if buyInstance:
            buyList.append(buyInstance)
            logger.info(f"Purchase made on {dataset.index[-1]}")
        else:
            logger.info(f"No purchase made for {company} on {dataset.index[-1]}")

def traderSell(dataset):
    closeToday = dataset['Close'].iloc[-1]
    for buy in buyList:
        if buy.date == dataset.index[-1]:
            logger.info(f"Skipping sale for {buy.companyName} as purchase was made on the same day.")
            continue
        else:
            dif = closeToday - buy.purchasePrice
            if dif > 5:
                sell(buy, .75)
            elif dif > 0 and dif <= 5:
                sell(buy, 0.5)

def buyListClean(buyList):
    return [buy for buy in buyList if buy.quantity > 0]

def tradeLoop(dataset, buyList, company, datasetWhole):
    datasetWeek = weekSort(dataset)
    fridays = datasetWeek.index[datasetWeek.index.weekday == 4]
    
    full_data = datasetWhole.copy()
    
    for friday in fridays:
        full_data_until_friday = full_data.loc[:friday]
        
        if len(full_data_until_friday) < 60:
            logger.warning(f"Insufficient data until {friday} for full_data_until_friday")
            continue

        buyList = buyListClean(buyList)
        
        traderBuy(full_data_until_friday, company)
        
        traderSell(full_data_until_friday)

# Example usage
companyName = input("Enter company ticker symbol: ")
model, dataset, datasetWhole = loadDatasetAndModel(companyName)

if model is not None and not dataset.empty:
    tradeLoop(dataset, buyList, companyName, datasetWhole)
else:
    logger.error("Error loading model or dataset. Exiting...")

# Calculate total assets and sum after trading
currentAsset = 0
buyList = buyListClean(buyList)
for buy in buyList:
    print(buy.quantity * dataset['Close'].iloc[-1])

total = sum + currentAsset

# Output summary
print("\nTransaction Summary:")
for log in transaction_log:
    print(log)

print("\nSummary: Current Price", dataset['Close'].iloc[-1])
print(f"Current Asset Value: {currentAsset}")
print(f"Remaining Cash: {sum}")
print(f"Total Value: {total}")
