import numpy as np
import pandas as pd
import datetime
import MetaTrader5 as mt5



def get_data(currency):
    mt5.initialize()
    ohlc_data = pd.DataFrame(mt5.copy_rates_range(currency, mt5.TIMEFRAME_M15, datetime.datetime(2015, 1, 1), datetime.datetime.now()))
    ohlc_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    ohlc_data.dropna(inplace=True)

    return ohlc_data
def bolinger_function(data,std,tpw,s_num):
    k = std 
    df=data
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3

    df['std_dev'] = df['typical_price'].rolling(window=tpw).std()

    df['upper_band'] = df[f'sma_{s_num}'] + k * df['std_dev']
    df['lower_band'] = df[f'sma_{s_num}'] - k * df['std_dev']
    return df

def isred(close,op):
    isre=None
    if close<op:
        isre=True
    return isre

def isgreen(close,op):
    isgree=None
    if close>op:
        isgree=True
    return isgree

def splitdf(data,size):
    splitdata=np.array_split(data, size)
    
    return splitdata
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

    
def evaluate(ls,name):
    df=pd.DataFrame(ls, columns=['Tradetype', 'dateopened', 'dateclosed', 'profit'])
    maxprof=df.profit.max()
    index_of_maxprof = df['profit'].idxmax()
    pips=maxprof*100
    totalprof=df.profit.sum()
    # print(f'The total profit for  {name} was :{totalprof}')
    df['dateopened'] = df['dateopened'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df['dateclosed'] = df['dateclosed'].apply(lambda x: datetime.datetime.fromtimestamp(x))

# Calculate the duration of each trade
    df['duration'] = df['dateclosed'] - df['dateopened']
    max_duration = df['duration'].max()
    id_max_duration=df['duration'].idxmax()
    max_row = df[df['duration'] == max_duration]
    longtype=max_row['Tradetype'].values[0]
    num_buy_trades = df[df['Tradetype'] == 'Buy'].shape[0]

# Finding the total profit for 'Buy' trades
    total_sell_profit = df[df['Tradetype'] == 'Sell']['profit'].sum()
    num_sell_trades = df[df['Tradetype'] == 'Sell'].shape[0]

# Finding the total profit for 'Buy' trades
    total_buy_profit = df[df['Tradetype'] == 'Buy']['profit'].sum()

    max_profit = max_row['profit'].values[0]
    # print(f'the trade that took most time took {max_duration} and made {max_profit} type:{longtype}')
    # print(f'Buy :{num_buy_trades} profit:{total_buy_profit}')
    # print(f'Sell:{num_sell_trades} profit:{total_sell_profit}')
    # print(f'the max duration was {max_duration}')
    # print(f'The trade with the highest profit made {maxprof} pips:{pips} ')
    profitable_trades = df[df['profit'] > 0].shape[0]
    losing_trades = df[df['profit'] < 0].shape[0]
    total_profit = df[df['profit'] > 0]['profit'].sum()
    total_loss = df[df['profit'] < 0]['profit'].sum()

    details=[totalprof,num_buy_trades,total_buy_profit,num_sell_trades,total_sell_profit,maxprof,index_of_maxprof,max_duration,id_max_duration,profitable_trades,losing_trades,total_profit,total_loss]
    return details



def calc_prof(enter,close):
    prof=enter-close
    return prof

def m_get_data(currency, timeframes, start_date, end_date):
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