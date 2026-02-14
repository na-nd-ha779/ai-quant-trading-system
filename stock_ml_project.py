import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Quant Trading System Started")

stock = input("Enter Stock Symbol (AAPL, MSFT, TSLA etc): ").strip().upper()

if stock == "":
    print("❌ Stock symbol cannot be empty!")
    exit()

# -----------------------------
# 1️⃣ Download Data (FIXED VERSION)
# -----------------------------
data = yf.download(stock, period="2y", auto_adjust=True)

if data.empty:
    print("❌ Invalid stock symbol or no data found!")
    exit()

# Flatten MultiIndex if exists
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Keep only needed columns
data = data[['Close', 'Volume']].copy()

# Force numeric (IMPORTANT FIX)
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------

# RSI
delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12 = data['Close'].ewm(span=12, adjust=False).mean()
ema26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26

# Bollinger Bands
data['MA20'] = data['Close'].rolling(20).mean()
data['BB_upper'] = data['MA20'] + 2 * data['Close'].rolling(20).std()
data['BB_lower'] = data['MA20'] - 2 * data['Close'].rolling(20).std()

# Volatility
data['Volatility'] = data['Close'].pct_change().rolling(20).std()

# Volume change
data['Volume_Change'] = data['Volume'].pct_change()

# Target
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

data = data.dropna()

# -----------------------------
# 3️⃣ ML Model
# -----------------------------
features = ['RSI', 'MACD', 'MA20', 'BB_upper', 'BB_lower',
            'Volatility', 'Volume_Change']

X = data[features]
y = data['Target']

split = int(len(data) * 0.7)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nML Direction Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# 4️⃣ Backtesting with Risk Management
# -----------------------------

initial_cash = 100000
cash = initial_cash
shares = 0
position = 0
entry_price = 0

risk_per_trade = 0.2
stop_loss_pct = 0.03
take_profit_pct = 0.06

portfolio_values = []
buy_hold_values = []

test_prices = data['Close'][split:]
buy_hold_shares = initial_cash / test_prices.iloc[0]

for i in range(len(predictions) - 1):

    today_price = test_prices.iloc[i]
    tomorrow_price = test_prices.iloc[i + 1]
    signal = predictions[i]

    if signal == 1 and position == 0:
        invest_amount = cash * risk_per_trade
        shares = invest_amount / tomorrow_price
        cash -= invest_amount
        entry_price = tomorrow_price
        position = 1

    if position == 1:
        change_pct = (tomorrow_price - entry_price) / entry_price

        if change_pct <= -stop_loss_pct or change_pct >= take_profit_pct or signal == 0:
            cash += shares * tomorrow_price
            shares = 0
            position = 0

    portfolio_value = cash + shares * tomorrow_price
    portfolio_values.append(portfolio_value)

    buy_hold_value = buy_hold_shares * tomorrow_price
    buy_hold_values.append(buy_hold_value)

final_value = portfolio_values[-1]
profit = final_value - initial_cash
profit_percent = (profit / initial_cash) * 100

buy_hold_final = buy_hold_values[-1]
buy_hold_profit = buy_hold_final - initial_cash
buy_hold_percent = (buy_hold_profit / initial_cash) * 100

print(f"\nAI Strategy Final Value: ₹{final_value:.2f}")
print(f"AI Strategy Return: {profit_percent:.2f}%")

print(f"\nBuy & Hold Final Value: ₹{buy_hold_final:.2f}")
print(f"Buy & Hold Return: {buy_hold_percent:.2f}%")

# -----------------------------
# 5️⃣ Plot
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(portfolio_values, label="AI Strategy")
plt.plot(buy_hold_values, label="Buy & Hold")
plt.title("Strategy vs Buy & Hold")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()

# -----------------------------
# 6️⃣ Performance Metrics
# -----------------------------

portfolio_series = pd.Series(portfolio_values)
daily_returns = portfolio_series.pct_change().dropna()

sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

rolling_max = portfolio_series.cummax()
drawdown = portfolio_series / rolling_max - 1
max_drawdown = drawdown.min()

years = 2 * 0.3
cagr = ((portfolio_series.iloc[-1] / initial_cash) ** (1/years)) - 1

print("\n===== PERFORMANCE REPORT =====")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")
print(f"CAGR: {cagr*100:.2f}%")
