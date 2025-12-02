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

st.markdown('''
# 📊Crypto Price and Prediction App
**Credits**
- App built by [Thirumala Sai Harsha Inumarthi](https://www.linkedin.com/in/saiharsha3377)
''')
st.write('---')

pd.options.display.float_format = '${:,.2f}'.format
load_dotenv()

symbol_map = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "ATOMUSDT": "ATOM-USD",
    "SOLUSDT": "SOL-USD",
    "ADAUSDT": "ADA-USD",
    "DOTUSDT": "DOT-USD",
    "MATICUSDT": "MATIC-USD",
    "AVAXUSDT": "AVAX-USD",
    "NEARUSDT": "NEAR-USD",
    "AAVEUSDT": "AAVE-USD",
    "FTMUSDT": "FTM-USD",
    "RUNEUSDT": "RUNE-USD"
}

st.sidebar.header('Query Parameters Price')
price_ticker = st.sidebar.selectbox('Ticker', list(symbol_map.keys()))
yf_symbol = symbol_map[price_ticker]

df = yf.download(yf_symbol, period="1d", interval="1m")
current_price = df["Close"].iloc[-1]
previous_close = df["Close"].iloc[-2]
delta = current_price - previous_close

st.metric(
    label=price_ticker,
    value=float(current_price),
    delta=float(delta)
)

df_candle = yf.download(yf_symbol, period="2y")

df_candle.reset_index(inplace=True)
df_candle.set_index("Date", inplace=True)

st.subheader(f'{price_ticker} Price Dataframe')
st.write(df_candle.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_candle.index, y=df_candle["Close"]))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_raw_data_log():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_candle.index, y=df_candle["Close"]))
    fig.update_yaxes(type="log")
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_bb_data():
    temp = df_candle.copy()
    temp["Open"] = temp["Open"]
    temp["High"] = temp["High"]
    temp["Low"] = temp["Low"]
    temp["Close"] = temp["Close"]
    qf = cf.QuantFig(temp, legend='top', name='Crypto')
    qf.add_bollinger_bands()
    qf.add_ema(periods=[12, 26, 200])
    qf.add_volume()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

options_klines = st.multiselect('Customize Charts', ['log', 'raw', 'bb_ema'])

if len(options_klines) == 0:
    st.subheader(f'{price_ticker} Price Area Chart')
    express = px.area(df_candle, x=df_candle.index, y='Close')
    st.write(express)

for choice in options_klines:
    if choice == 'log':
        plot_raw_data_log()
    if choice == 'raw':
        plot_raw_data()
    if choice == 'bb_ema':
        plot_bb_data()

st.sidebar.header('Query Parameters Prediction')
prediction_ticker = st.sidebar.selectbox('Prediction Ticker', list(symbol_map.values()))

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
