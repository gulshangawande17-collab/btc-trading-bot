import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Pull hidden keys from the server environment
API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

print("Fetching data & calculating features...")
btc_ticker = yf.Ticker("BTC-USD")
btc_data = btc_ticker.history(period="2y")
btc_data.index = btc_data.index.tz_localize(None)

btc_data['Daily_Return'] = btc_data['Close'].pct_change()
btc_data['SMA_10'] = btc_data['Close'].rolling(window=10).mean()
btc_data['SMA_50'] = btc_data['Close'].rolling(window=50).mean()
btc_data['Volatility'] = btc_data['Daily_Return'].rolling(window=10).std()

delta = btc_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
btc_data['RSI_14'] = 100 - (100 / (1 + rs))

exp1 = btc_data['Close'].ewm(span=12, adjust=False).mean()
exp2 = btc_data['Close'].ewm(span=26, adjust=False).mean()
btc_data['MACD'] = exp1 - exp2
btc_data['MACD_Signal'] = btc_data['MACD'].ewm(span=9, adjust=False).mean()

btc_data['Target'] = np.where(btc_data['Close'].shift(-1) > btc_data['Close'], 1, 0)
btc_data = btc_data.dropna()

print("Training Neural Network...")
features = ['Daily_Return', 'SMA_10', 'SMA_50', 'Volatility', 'RSI_14', 'MACD', 'MACD_Signal']
X = btc_data[features]
y = btc_data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, activation='relu')
model.fit(X_scaled, y)

todays_data = X.iloc[-1:]
todays_data_scaled = scaler.transform(todays_data)
prediction = model.predict(todays_data_scaled)[0]

print(f"Prediction: {'BUY' if prediction == 1 else 'SELL'}")

symbol = 'BTC/USD'
qty_to_trade = 0.1

try:
    position = trading_client.get_open_position(symbol)
    current_qty = float(position.qty)
except Exception:
    current_qty = 0.0

if prediction == 1 and current_qty == 0:
    print("Executing BUY...")
    market_order_data = MarketOrderRequest(symbol=symbol, qty=qty_to_trade, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
    trading_client.submit_order(order_data=market_order_data)
    print("BUY order submitted!")
elif prediction == 0 and current_qty > 0:
    print("Executing SELL...")
    market_order_data = MarketOrderRequest(symbol=symbol, qty=current_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
    trading_client.submit_order(order_data=market_order_data)
    print("SELL order submitted!")
else:
    print("HOLDING current position.")
