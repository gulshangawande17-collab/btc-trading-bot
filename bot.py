import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
btc_data = btc_ticker.history(period="5d", interval="15m")
btc_data.index = btc_data.index.tz_localize(None)

print("2. Calculating Intraday Technical Indicators (6-Hour Lookback)...")
btc_data['EMA_9'] = btc_data['Close'].ewm(span=9, adjust=False).mean()
btc_data['EMA_21'] = btc_data['Close'].ewm(span=21, adjust=False).mean()

btc_data['Support_6h'] = btc_data['Low'].rolling(window=24).min()
btc_data['Resistance_6h'] = btc_data['High'].rolling(window=24).max()
btc_data['Dist_to_Support'] = btc_data['Close'] - btc_data['Support_6h']
btc_data['Dist_to_Resistance'] = btc_data['Resistance_6h'] - btc_data['Close']

btc_data['Volume_Avg_6h'] = btc_data['Volume'].rolling(window=24).mean()
btc_data['Volume_Surge'] = btc_data['Volume'] / btc_data['Volume_Avg_6h'] 

typical_price = (btc_data['High'] + btc_data['Low'] + btc_data['Close']) / 3
btc_data['VWAP'] = (typical_price * btc_data['Volume']).cumsum() / btc_data['Volume'].cumsum()

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

print("4. Generating Live Intraday Signal & Risk Parameters...")
todays_data = X.iloc[-1:]
todays_data_scaled = scaler.transform(todays_data)
prediction = model.predict(todays_data_scaled)[0]

# --- NEW: CALCULATE SL AND TARGET ---
current_price = btc_data['Close'].iloc[-1]
current_support = btc_data['Support_6h'].iloc[-1]
current_resistance = btc_data['Resistance_6h'].iloc[-1]

if prediction == 1: # BUY setup
    sl = current_support * 0.995 # SL is 0.5% below the 6h floor
    tp = current_resistance if current_resistance > current_price else current_price * 1.015
    decision = "BUY"
else: # SELL setup
    sl = current_resistance * 1.005 # SL is 0.5% above the 6h ceiling
    tp = current_support if current_support < current_price else current_price * 0.985
    decision = "SELL"

print("=========================================")
print(f"AI PREDICTION : {decision}")
print(f"CURRENT PRICE : ${current_price:.2f}")
print(f"TAKE PROFIT   : ${tp:.2f}")
print(f"STOP LOSS     : ${sl:.2f}")
print("=========================================")

# --- NEW: GENERATE AND SAVE THE CHART PICTURE ---
print("5. Drawing the Analysis Chart...")
plt.figure(figsize=(12, 6))
# Plot only the last 50 candles (approx 12 hours) so the chart isn't too squished
plot_data = btc_data.iloc[-50:] 

plt.plot(plot_data.index, plot_data['Close'], label='BTC Price', color='black', linewidth=2)
plt.plot(plot_data.index, plot_data['EMA_9'], label='EMA 9 (Fast)', color='blue', linestyle='--')
plt.plot(plot_data.index, plot_data['EMA_21'], label='EMA 21 (Slow)', color='orange', linestyle='--')
plt.axhline(y=current_support, color='red', linestyle='-', alpha=0.5, label=f'Support (${current_support:.0f})')
plt.axhline(y=current_resistance, color='green', linestyle='-', alpha=0.5, label=f'Resistance (${current_resistance:.0f})')

plt.title(f"AI Intraday Analysis | Decision: {decision} | Target: ${tp:.0f} | SL: ${sl:.0f}")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the picture to the server
plt.savefig('chart.png', bbox_inches='tight')
print("Chart successfully saved as chart.png!")

# --- EXECUTION LOGIC ---
symbol = 'BTC/USD'
qty_to_trade = 0.1

try:
    position = trading_client.get_open_position(symbol)
    current_qty = float(position.qty)
except Exception:
    current_qty = 0.0

if prediction == 1 and current_qty == 0:
    print("Executing BUY...")
    trading_client.submit_order(order_data=MarketOrderRequest(symbol=symbol, qty=qty_to_trade, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
elif prediction == 0 and current_qty > 0:
    print("Executing SELL...")
    trading_client.submit_order(order_data=MarketOrderRequest(symbol=symbol, qty=current_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC))
else:
    print("HOLDING current position.")
