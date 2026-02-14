import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("🌳 AI-Based Quant Trading System (Random Forest)")

# -------------------------------
# Load Data
# -------------------------------
def load_data(symbol):
    data = yf.download(symbol, period="2y")
    data['Returns'] = data['Close'].pct_change()
    return data


# -------------------------------
# Add Indicators
# -------------------------------
def add_indicators(data):

    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()

    # Extra Features
    data['Volatility'] = data['Returns'].rolling(10).std()
    data['Momentum'] = data['Close'] - data['Close'].shift(10)

    return data


# -------------------------------
# ML Strategy
# -------------------------------
def ml_strategy(data):

    data = data.dropna().copy()

    data['Target'] = np.where(data['Returns'].shift(-1) > 0, 1, 0)

    features = ['RSI', 'MA50', 'MA200', 'Returns', 'Volatility', 'Momentum']
    X = data[features]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    # ✅ Save index BEFORE scaling
    test_index = X_test.index

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # ✅ Use saved index
    data.loc[test_index, 'Prediction'] = predictions
    data['Position'] = data['Prediction']
    data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']

    return data, accuracy

   

# -------------------------------
# Performance Metrics
# -------------------------------
def calculate_metrics(returns):

    returns = returns.fillna(0)

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility != 0 else 0

    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = drawdown.min()

    return total_return, sharpe, max_dd


# -------------------------------
# UI
# -------------------------------
symbol = st.text_input("Enter Stock Symbol", "MSFT")

if st.button("Run Strategy"):

    data = load_data(symbol)

    if data.empty:
        st.error("Invalid Stock Symbol")
    else:

        data = add_indicators(data)
        data, accuracy = ml_strategy(data)

        data['Market_Cumulative'] = (1 + data['Returns']).cumprod()
        data['Strategy_Cumulative'] = (1 + data['Strategy_Returns'].fillna(0)).cumprod()

        market_metrics = calculate_metrics(data['Returns'])
        strategy_metrics = calculate_metrics(data['Strategy_Returns'])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Equity Curve")
            fig, ax = plt.subplots()
            ax.plot(data['Market_Cumulative'], label="Buy & Hold")
            ax.plot(data['Strategy_Cumulative'], label="ML Strategy")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Performance Metrics")
            st.success(f"Accuracy: {round(accuracy*100,2)} %")
            st.write("ML Strategy Return:", round(strategy_metrics[0]*100,2), "%")
            st.write("Sharpe Ratio:", round(strategy_metrics[1],2))
            st.write("Max Drawdown:", round(strategy_metrics[2]*100,2), "%")
