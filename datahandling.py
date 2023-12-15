import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from functions import isred,isgreen
import itertools

def get_data(currency):
    ohlc_data = pd.DataFrame(mt5.copy_rates_range(currency, mt5.TIMEFRAME_M15, datetime(2021, 1, 1), datetime.now()))
    ohlc_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    ohlc_data.dropna(inplace=True)

    return ohlc_data

def rsi( data, periods):
    for period in periods:
        # Calculate the change in price
        delta = data['Close'].diff()

        # Make gains and losses series
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        # Calculate the average gain and average loss
        avg_gain = up.rolling(window=period).mean()
        avg_loss = down.rolling(window=period).mean()

        # Calculate the relative strength
        rs = avg_gain / avg_loss

        # Calculate the RSI
        rsi = 100 - (100 / (1 + rs))

        # Add the RSI values as a new column with period number
        data['rsi_{}'.format(period)] = rsi
    return data
def calculate_sma(data, periods):
    for period in periods:
        sma_values = []
        close_prices = data['Close'].values

        for i in range(period, len(data) + 1):
            sma = sum(close_prices[i - period:i]) / period
            sma_values.append(sma)

            data[f'sma_{period}'] = [None] * (period - 1) + sma_values

        return data
def bolinger(self, data, std_list, tpw_list, s_num_list):
    result_dfs = []

    for std, tpw, s_num in itertools.product(std_list, tpw_list, s_num_list):
        k = std  # Number of standard deviations

            # Calculate typical price
        data['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3

            # Calculate standard deviation
        data['std_dev'] = data['typical_price'].rolling(window=tpw).std()  # Assuming a window of 5 periods

            # Calculate Bollinger Bands using the existing 'sma' column
        data[f'upper_band_{std}_{tpw}_{s_num}'] = data[f'sma_{s_num}'] + k * data['std_dev']
        data[f'lower_band_{std}_{tpw}_{s_num}'] = data[f'sma_{s_num}'] - k * data['std_dev']

        result_dfs.append(data.copy())

    return result_dfs