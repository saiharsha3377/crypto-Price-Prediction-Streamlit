import streamlit as st
import pandas as pd
import yfinance as yf
import cufflinks as cf
from dotenv import load_dotenv
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import requests

st.markdown('''
# 📊Crypto Price and Prediction App
**Credits**
- App built by [Thirumala Sai Harsha Inumarthi](https://www.linkedin.com/in/saiharsha3377)
''')
st.write('---')

pd.options.display.float_format = '${:,.2f}'.format
load_dotenv()

coin_map = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
    "ATOMUSDT": "atom",
    "SOLUSDT": "sol",
    "ADAUSDT": "ada",
    "DOTUSDT": "dot",
    "MATICUSDT": "matic",
    "AVAXUSDT": "avax",
    "NEARUSDT": "near",
    "AAVEUSDT": "aave",
    "FTMUSDT": "ftm",
    "RUNEUSDT": "rune"
}

def load_prices():
    url = "https://raw.githubusercontent.com/coinapi-data/market-data/main/prices.json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["id"] = df["symbol"].str.lower()
    return df

df = load_prices()

st.sidebar.header('Query Parameters Price')
price_ticker = st.sidebar.selectbox('Ticker', list(coin_map.keys()))

token = coin_map[price_ticker]
row = df[df["id"] == token]

st.metric(
    label=price_ticker,
    value=float(row["price"]),
    delta=str(float(row["change_24h"])) + "%"
)

def load_candles(symbol):
    url = f"https://raw.githubusercontent.com/coinapi-data/market-data/main/{symbol}-ohlcv.json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    d["Date"] = pd.to_datetime(d["time"])
    d.set_index("Date", inplace=True)
    return d

klines_ticker_price = load_candles(token)

st.subheader(f'{price_ticker} Price Dataframe')
st.write(klines_ticker_price.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=klines_ticker_price.index, y=klines_ticker_price["close"]))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_raw_data_log():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=klines_ticker_price.index, y=klines_ticker_price["close"]))
    fig.update_yaxes(type="log")
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_bb_data():
    temp = klines_ticker_price.copy()
    temp["Open"] = temp["close"]
    temp["High"] = temp["close"]
    temp["Low"] = temp["close"]
    temp["Close"] = temp["close"]
    qf = cf.QuantFig(temp, legend='top', name='Crypto')
    qf.add_bollinger_bands()
    qf.add_ema(periods=[12, 26, 200])
    qf.add_volume()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

options_klines = st.multiselect('Customize Charts', ['log', 'raw', 'bb_ema'])
if len(options_klines) == 0:
    st.subheader(f'{price_ticker} Price Area Chart')
    express = px.area(klines_ticker_price, x=klines_ticker_price.index, y='close')
    st.write(express)

for choice in options_klines:
    if choice == 'log':
        plot_raw_data_log()
    if choice == 'raw':
        plot_raw_data()
    if choice == 'bb_ema':
        plot_bb_data()

st.sidebar.header('Query Parameters Prediction')
prediction_ticker = st.sidebar.selectbox('Prediction Ticker',
    ('BTC-USD', 'ETH-USD', 'ATOM-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD',
     'MATIC-USD', 'AVAX-USD', 'NEAR-USD', 'AAVE-USD', 'FTM-USD', 'RUNE-USD'))

start_date = st.sidebar.date_input("Start date", date(2016, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.today())

n_years = st.sidebar.slider("Years of prediction:", 1, 10)
n_days = st.sidebar.slider("Days of prediction:", 7, 90)
years_period = n_years * 365

df_yf = yf.download(prediction_ticker, start_date, end_date)
df_yf.reset_index(inplace=True)
df_yf['Close'] = pd.to_numeric(df_yf['Close'], errors='coerce')
df_yf = df_yf.dropna(subset=['Close'])

df_train = df_yf[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(seasonality_mode="multiplicative")
m.fit(df_train)
future_years = m.make_future_dataframe(periods=years_period)
future_days = m.make_future_dataframe(periods=n_days)
forecast_years = m.predict(future_years)
forecast_days = m.predict(future_days)

def plot_year_prediction():
    fig1 = plot_plotly(m, forecast_years)
    st.plotly_chart(fig1)

def plot_year_components():
    fig2 = m.plot_components(forecast_years)
    st.write(fig2)

def plot_day_prediction():
    fig3 = plot_plotly(m, forecast_days)
    st.plotly_chart(fig3)

def plot_day_components():
    fig4 = m.plot_components(forecast_days)
    st.write(fig4)

if st.button('Year Prediction Plot'):
    plot_year_prediction()
    plot_year_components()

if st.button('Days Prediction Plot'):
    plot_day_prediction()
    plot_day_components()
