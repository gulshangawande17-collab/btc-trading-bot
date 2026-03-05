import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

print("1. Fetching Intraday Data (15-Minute Intervals)...")
btc_ticker = yf.Ticker("BTC-USD")
# Fetch the last 5 days of data, broken down into 15-minute candles
btc_data = btc_ticker.history(period="5d", interval="15m")
btc_data.index = btc_data.index.tz_localize(None)

print("2. Calculating Intraday Technical Indicators (6-Hour Lookback)...")
# A. TREND: Fast Exponential Moving Averages
btc_data['EMA_9'] = btc_data['Close'].ewm(span=9, adjust=False).mean()
btc_data['EMA_21'] = btc_data['Close'].ewm(span=21, adjust=False).mean()

# B. SUPPORT & RESISTANCE: Highs and Lows of the last 6 hours (24 periods of 15 mins)
btc_data['Support_6h'] = btc_data['Low'].rolling(window=24).min()
btc_data['Resistance_6h'] = btc_data['High'].rolling(window=24).max()
# Teach the AI how close the current price is to these critical levels
btc_data['Dist_to_Support'] = btc_data['Close'] - btc_data['Support_6h']
btc_data['Dist_to_Resistance'] = btc_data['Resistance_6h'] - btc_data['Close']

# C. VOLUME & LIQUIDITY: Volume Surge and VWAP
btc_data['Volume_Avg_6h'] = btc_data['Volume'].rolling(window=24).mean()
# If this is > 1, volume is surging (liquidity is entering)
btc_data['Volume_Surge'] = btc_data['Volume'] / btc_data['Volume_Avg_6h'] 

# Approximate VWAP (Typical Price * Volume)
typical_price = (btc_data['High'] + btc_data['Low'] + btc_data['Close']) / 3
btc_data['VWAP'] = (typical_price * btc_data['Volume']).cumsum() / btc_data['Volume'].cumsum()

# TARGET: Will the next 15-minute candle close higher?
btc_data['Target'] = np.where(btc_data['Close'].shift(-1) > btc_data['Close'], 1, 0)
btc_data = btc_data.dropna()

print("3. Training the AI on the new Intraday Features...")
features = ['EMA_9', 'EMA_21', 'Dist_to_Support', 'Dist_to_Resistance', 'Volume_Surge', 'VWAP']
X = btc_data[features]
y = btc_data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, activation='relu')
model.fit(X_scaled, y)

print("4. Generating Live Intraday Signal...")
todays_data = X.iloc[-1:]
todays_data_scaled = scaler.transform(todays_data)
prediction = model.predict(todays_data_scaled)[0]

print(f"Intraday Prediction for next 15 mins: {'BUY (UPTREND)' if prediction == 1 else 'SELL (DOWNTREND)'}")

# Execution Logic
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
    print("Executing SELL to take profit/cut losses...")
    market_order_data = MarketOrderRequest(symbol=symbol, qty=current_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
    trading_client.submit_order(order_data=market_order_data)
    print("SELL order submitted!")
else:
    print("HOLDING current position.")
