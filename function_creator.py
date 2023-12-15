import numpy as np
import scipy
import MetaTrader5 as mt5
import pandas as pd

# def find_levels (price: np.array, atr: float,first_w: float = 0.1,atr_mult: float = 3.0,prom_thresh: float = 0.1): # Log closing price
#     w = 1.0
#     w_step= (last w first_w) / len(price)
#     weights = first_w + np.arange (len (price))*w_step
#     weights [weights < 0] = 0.0
#     kernal = scipy.stats.gaussian_kde (price, bw_method=
#     min_v = np.min(price)
#     = np.max (price)
#     = (max_v - min_v) / 200
#     price_range= np.arange (min_v, max_v, step)
#     = kernal(price range) # Market profile
#     Find significant peaks in the market profile
#     pdf_max = np.max (pdf)
#     prom_min= pdf_max * prom_thresh
#     peaks, props = scipy.signal.find_peaks (pdf, prominer
#     levels = []
#     peak in peaks:
#     levels.append(np.exp(price_range [peak]))
#     return levels, peaks, props, price range, pdf, weigh

def get_data(currency, timeframes, start_date, end_date):
    mt5.initialize()

    # Convert start_date and end_date to datetime objects if they are not already
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Define the timeframe mapping
    timeframe_mapping = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        # 'M30':mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1,
        # Add more timeframes as needed
    }

    # Check if the provided timeframes are valid
    invalid_timeframes = set(timeframes) - set(timeframe_mapping.keys())
    if invalid_timeframes:
        raise ValueError(f"Invalid timeframes: {', '.join(invalid_timeframes)}")

    ohlc_data_dict = {}

    # Fetch data for each specified timeframe
    for timeframe in timeframes:
        ohlc_data = pd.DataFrame(mt5.copy_rates_range(currency, timeframe_mapping[timeframe], start_date, end_date))

        # Rename columns for consistency
        ohlc_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)

        # Drop NaN values, if any
        ohlc_data.dropna(inplace=True)

        ohlc_data_dict[timeframe.lower()] = ohlc_data

    return ohlc_data_dict