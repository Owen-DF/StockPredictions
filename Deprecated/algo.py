import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def simulate_trading(data, model, initial_balance=10000, shares=0, look_back=60):
    X, _, scaler = prepare_data(data, look_back)
    
    balance = initial_balance
    buy_hold_balance = initial_balance
    buy_hold_shares = initial_balance // data['Close'].iloc[look_back]
    
    trades = []
    
    for i in range(len(X)):
        current_price = data['Close'].iloc[i+look_back]
        current_date = data.index[i+look_back]
        
        # Make prediction
        prediction = model.predict(X[i].reshape(1, look_back, 1))
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        
        # Simple trading strategy
        if predicted_price > current_price * 1.01 and balance > current_price:  # Buy
            shares_to_buy = balance // current_price
            balance -= shares_to_buy * current_price
            shares += shares_to_buy
            trades.append(('BUY', shares_to_buy, current_price, current_date))
        elif predicted_price < current_price * 0.99 and shares > 0:  # Sell
            balance += shares * current_price
            trades.append(('SELL', shares, current_price, current_date))
            shares = 0
        
        # Update buy and hold strategy
        buy_hold_balance = buy_hold_shares * current_price
    
    # Sell any remaining shares at the end
    if shares > 0:
        final_price = data['Close'].iloc[-1]
        final_date = data.index[-1]
        balance += shares * final_price
        trades.append(('SELL', shares, final_price, final_date))
    
    return balance, buy_hold_balance, trades

# Main execution
ticker = input("Enter company ticker symbol: ")

# load three months and 30 days
end = datetime.date.today()
start = end - datetime.timedelta(days=180)

data = yf.download(ticker, start=start, end=end)

model_dict = {
    "IBM": "ExcelFiles/IBM/IBM.h5",
    "URI": "ExcelFiles/URI/URI.h5",
    "NVDA": "ExcelFiles/NVDA/NVDA.h5",
    "AAPL": "ExcelFiles/AAPL/AAPL.h5",
    "SIRI": "ExcelFiles/SIRI/SIRI.h5",
    "BA": "ExcelFiles/BA/BA.h5"
}

if ticker in model_dict:
    model = load_model(model_dict[ticker])
else:
    print(f"No pre-trained model available for {ticker}")
    exit()

# Run simulation
final_balance, buy_hold_balance, trades = simulate_trading(data, model)

# Print results
print(f"Final balance: ${final_balance:.2f}")
print(f"Buy and hold balance: ${buy_hold_balance:.2f}")
print(f"Number of trades: {len(trades)}")
print("\nTrade log:")
for trade in trades:
    print(f"{trade[0]}: {trade[1]} shares at ${trade[2]:.2f} on {trade[3].strftime('%Y-%m-%d')}")

# Calculate and print performance metrics
initial_balance = 10000
algo_return = (final_balance - initial_balance) / initial_balance * 100
buy_hold_return = (buy_hold_balance - initial_balance) / initial_balance * 100

print(f"\nAlgorithm return: {algo_return:.2f}%")
print(f"Buy and hold return: {buy_hold_return:.2f}%")