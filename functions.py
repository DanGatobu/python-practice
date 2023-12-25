import numpy as np
import pandas as pd
import datetime
import MetaTrader5 as mt5
from enum import Enum
import csv

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

def evaluate2(ls, name):
    df = pd.DataFrame(ls, columns=['Tradetype', 'dateopened', 'dateclosed', 'profit'])
    maxprof = df.profit.max()
    index_of_maxprof = df['profit'].idxmax()
    pips = maxprof * 100
    totalprof = df.profit.sum()

    df['dateopened'] = df['dateopened'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df['dateclosed'] = df['dateclosed'].apply(lambda x: datetime.datetime.fromtimestamp(x))

    # Calculate the duration of each trade
    df['duration'] = df['dateclosed'] - df['dateopened']
    max_duration = df['duration'].max()
    id_max_duration = df['duration'].idxmax()
    max_row = df[df['duration'] == max_duration]
    longtype = max_row['Tradetype'].values[0]
    num_buy_trades = df[df['Tradetype'] == 'Buy'].shape[0]

    # Finding the total profit for 'Buy' trades
    total_sell_profit = df[df['Tradetype'] == 'Sell']['profit'].sum()
    num_sell_trades = df[df['Tradetype'] == 'Sell'].shape[0]

    # Finding the total profit for 'Buy' trades
    total_buy_profit = df[df['Tradetype'] == 'Buy']['profit'].sum()

    max_profit = max_row['profit'].values[0]

    profitable_trades = df[df['profit'] > 0].shape[0]
    losing_trades = df[df['profit'] < 0].shape[0]
    total_profit = df[df['profit'] > 0]['profit'].sum()
    total_loss = df[df['profit'] < 0]['profit'].sum()

    details = [totalprof, num_buy_trades, total_buy_profit, num_sell_trades, total_sell_profit, maxprof,
               index_of_maxprof, max_duration, id_max_duration, profitable_trades, losing_trades, total_profit,
               total_loss]

    # Create a DataFrame with a single row using transposition
    result_df = pd.DataFrame(details).T
    result_df.columns = ['totalprof', 'num_buy_trades', 'total_buy_profit', 'num_sell_trades', 'total_sell_profit',
                         'maxprof', 'index_of_maxprof', 'max_duration', 'id_max_duration', 'profitable_trades',
                         'losing_trades', 'total_profit', 'total_loss']

    return result_df

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

def find_prev_main3(data, window, index):
    df = data

    # Convert 'High' and 'Low' columns to NumPy arrays for slicing
    high_points = df['High'].values[max(0, index - window):index + 1]
    low_points = df['Low'].values[max(0, index - window):index + 1]

    # Vectorized operation to find smallest and largest values
    support = np.partition(low_points, 3)[:3].tolist()
    resistance = np.partition(high_points, -3)[-3:].tolist()

    return resistance, support


def mfind_tp(trade_type, resistance, support, price):
    first_high = resistance[0]
    second_high = resistance[1]
    third_high = resistance[2]
    first_low = support[0]
    second_low = support[1]
    third_low = support[2]
    
    level1 = {'trade_type': trade_type, 'ratio': 0, 'sl_price': 0, 'tp_price': 0, 'expected_profit': 0, 'expected_loss': 0}
    level2 = {'trade_type': trade_type, 'ratio': 0, 'sl_price': 0, 'tp_price': 0, 'expected_profit': 0, 'expected_loss': 0}
    level3 = {'trade_type': trade_type, 'ratio': 0, 'sl_price': 0, 'tp_price': 0, 'expected_profit': 0, 'expected_loss': 0}
    
    if trade_type == 'Buy':
        first_degree_low = price - first_low
        first_degree_high = first_high - price
        
        if first_degree_high < 0:
            level1 = {'trade_type': trade_type, 'ratio': 0, 'sl_price': first_low, 'tp_price': np.nan, 'expected_profit': first_degree_high, 'expected_loss': first_degree_low}
        else:
            ratio = first_degree_high / first_degree_low
            level1 = {'trade_type': trade_type, 'ratio': ratio, 'sl_price': first_low, 'tp_price': first_high, 'expected_profit': first_degree_high, 'expected_loss': first_degree_low}
        
        second_degree_low = price - second_low
        second_degree_high = second_high - price
        
        if second_degree_high < 0:
            level2 = level1
        else:
            ratio2 = second_degree_high / second_degree_low
            level2 = {'trade_type': trade_type, 'ratio': ratio2, 'sl_price': second_low, 'tp_price': second_high, 'expected_profit': second_degree_high, 'expected_loss': second_degree_low}
       
        third_degree_low = price - third_low
        third_degree_high = third_high - price
        
        if third_degree_high < 0:
            level3 = level2
        else:
            ratio3 = third_degree_high / third_degree_low
            level3 = {'trade_type': trade_type, 'ratio': ratio3, 'sl_price': third_low, 'tp_price': third_high, 'expected_profit': third_degree_high, 'expected_loss': third_degree_low}
        
    else:
        first_degree_low_sell = first_low - price
        first_degree_high_sell = price - first_low
        
        if first_degree_high_sell < 0: 
            level1 = {'trade_type': trade_type, 'ratio': 0, 'sl_price': first_low, 'tp_price': 'invalid', 'expected_profit': first_degree_high_sell, 'expected_loss': first_degree_low_sell}
        else:
            ratio = first_degree_high_sell / first_degree_low_sell
            level1 = {'trade_type': trade_type, 'ratio': ratio, 'sl_price': first_low, 'tp_price': first_high, 'expected_profit': first_degree_high_sell, 'expected_loss': first_degree_low_sell}
        
        second_degree_low_sell = second_low - price
        second_degree_high_sell = price - second_low
        
        if second_degree_high_sell < 0:
            level2 = level1
        else:
            ratio2 = second_degree_high_sell / second_degree_low_sell
            level2 = {'trade_type': trade_type, 'ratio': ratio2, 'sl_price': second_low, 'tp_price': second_high, 'expected_profit': second_degree_high_sell, 'expected_loss': second_degree_low_sell}
        
        third_degree_low_sell = third_high - price
        third_degree_high_sell = price - third_low
        
        if third_degree_high_sell < 0:
            level3 = level2
        else:
            ratio3 = third_degree_high_sell / third_degree_low_sell
            level3 = {'trade_type': trade_type, 'ratio': ratio3, 'sl_price': third_low, 'tp_price': third_high, 'expected_profit': third_degree_high_sell, 'expected_loss': third_degree_low_sell}
        
    return level1, level2, level3


def joined(using_data, win, small_sma, large_sma):
    dfs=smastrategy2(using_data, win, small_sma, large_sma)
    data_tp1=dfs[0]
    data_tp2=dfs[1]
    data_tp3=dfs[2]
    name=f'sma crossover {small_sma} & {large_sma} window{win}'
    evaluated_info=evaluate(data_tp1,name)
    evaluated_info2=evaluate(data_tp2,name)
    evaluated_info3=evaluate(data_tp3,name)
    result_dict1 = {
        'Strategy': name,
        'Take_profit_level':"level1(tp1)",
        'Max_Profit': evaluated_info[0],
        'Num_Buy_Trades': evaluated_info[1],
        'Total_Buy_Profit': evaluated_info[2],
        'Num_Sell_Trades': evaluated_info[3],
        'Total_Sell_Profit': evaluated_info[4],
        'Max_Profit_Index': evaluated_info[5],
        'Index_of_Max_Profit': evaluated_info[6],
        'Max_Duration': evaluated_info[7],
        'ID_Max_Duration': evaluated_info[8],
        'total_profitable_trades':evaluated_info[9],
        'total_unprofitable_trades':evaluated_info[10],
        'total_profitable_trades':evaluated_info[11],
        'total_loss_trades':evaluated_info[12]}
        
    result_dict2 = {
    'Strategy': name,
    'Take_profit_level':"level2(tp2)",
    'Max_Profit': evaluated_info2[0],
    'Num_Buy_Trades': evaluated_info2[1],
    'Total_Buy_Profit': evaluated_info2[2],
    'Num_Sell_Trades': evaluated_info2[3],
    'Total_Sell_Profit': evaluated_info2[4],
    'Max_Profit_Index': evaluated_info2[5],
    'Index_of_Max_Profit': evaluated_info2[6],
    'Max_Duration': evaluated_info2[7],
    'ID_Max_Duration': evaluated_info2[8],
    'total_profitable_trades':evaluated_info2[9],
    'total_unprofitable_trades':evaluated_info2[10],
    'total_profitable_trades':evaluated_info2[11],
    'total_loss_trades':evaluated_info2[12]}

    result_dict3 = {
    'Strategy': name,
    'Take_profit_level':"level3(tp3)",
    'Max_Profit': evaluated_info3[0],
    'Num_Buy_Trades': evaluated_info3[1],
    'Total_Buy_Profit': evaluated_info3[2],
    'Num_Sell_Trades': evaluated_info3[3],
    'Total_Sell_Profit': evaluated_info3[4],
    'Max_Profit_Index': evaluated_info3[5],
    'Index_of_Max_Profit': evaluated_info3[6],
    'Max_Duration': evaluated_info3[7],
    'ID_Max_Duration': evaluated_info3[8],
    'total_profitable_trades':evaluated_info3[9],
    'total_unprofitable_trades':evaluated_info3[10],
    'total_profitable_trades':evaluated_info3[11],
    'total_loss_trades':evaluated_info3[12]}
    return [result_dict1,result_dict2,result_dict3,dfs]
    
def smastrategy2(data, window, o_sma_value, s_sma_value):
    df = data
    trade = {}
    istrade = None
    istrade2 = None
    istrade3 = None

    openprice = 0
    tp1 = 0
    tp2 = 0
    tp3 = 0
    sl1 = 0
    sl2 = 0
    sl3 = 0
    opendate = 0
    processed_trades1 = []
    processed_trades2 = []
    processed_trades3 = []
    trade_class = None

    # Vectorized calculation of sma values
    sma_o_values = df[f'sma_{o_sma_value}'].values
    sma_s_values = df[f'sma_{s_sma_value}'].values

    for i in range(window, len(df)):
        if sma_o_values[i] > sma_s_values[i]:
            if istrade is None:
                trade_type = 'Buy'
                entryprice = df['Close'][i]
                resistance, support = find_prev_main3(data, window, i)
                t1, t2, t3 = mfind_tp(trade_type, resistance, support, entryprice)
                tp1 = t1['tp_price']
                tp2 = t2['tp_price']
                tp3 = t3['tp_price']
                sl1 = t1['sl_price']
                sl2 = t2['sl_price']
                sl3 = t3['sl_price']

                opendate = df['time'][i]
                istrade = 1
                istrade2 = 1
                istrade3 = 1
                if tp1 == 'invalid':
                    continue
                if tp1 == np.nan:
                    continue
                openprice = entryprice
                trade_class = trade_type
                
        elif sma_o_values[i] < sma_s_values[i]:
            if istrade is None:
                trade_type = 'Sell'
                trade_class = trade_type
                entryprice = df['Close'][i]
                opendate = df['time'][i]
                resistance, support = find_prev_main3(data, window, i)
                t1, t2, t3 = mfind_tp(trade_type, resistance, support, entryprice)
                openprice = entryprice
                tp1 = t1['tp_price']
                tp2 = t2['tp_price']
                tp3 = t3['tp_price']
                if tp1 == 'invalid':
                    continue
                if tp1 == np.nan:
                    continue
                istrade = 1
                istrade2 = 1
                istrade3 = 1
        
        price = data['Close'][i]
        if istrade == 1:
            if trade_class == 'Buy':
                if price > tp1 or price < sl1:
                    istrade = None
                    closedate = df['time'][i]
                    profit = calc_prof(price, openprice)
                    processed_trades1.append(('Buy', opendate, closedate, profit))
            if trade_class == 'Sell':
                if price < tp1 or price > sl1:
                    istrade = None
                    closedate = df['time'][i]
                    profit = calc_prof(openprice, price)
                    processed_trades1.append(('Sell', opendate, closedate, profit))
        if istrade2 == 1:
            if trade_class == 'Buy':
                if price > tp2 or price < sl2:
                    istrade2 = None
                    closedate = df['time'][i]
                    profit = calc_prof(price, openprice)
                    processed_trades2.append(('Buy', opendate, closedate, profit))
            if trade_class == 'Sell':
                if price < tp2 or price > sl2:
                    istrade2 = None
                    closedate = df['time'][i]
                    profit = calc_prof(openprice, price)
                    processed_trades2.append(('Sell', opendate, closedate, profit))
        if istrade3 == 1:
            if trade_class == 'Buy':
                if price > tp3 or price < sl3:
                    istrade3 = None
                    closedate = df['time'][i]
                    profit = calc_prof(price, openprice)
                    processed_trades3.append(('Buy', opendate, closedate, profit))
            if trade_class == 'Sell':
                if price < tp3 or price > sl3:
                    istrade3 = None
                    closedate = df['time'][i]
                    profit = calc_prof(openprice, price)
                    processed_trades3.append(('Sell', opendate, closedate, profit))
        
    processed_trades = [f'small{o_sma_value},large{s_sma_value}window{window}',processed_trades1, processed_trades2, processed_trades3]         

    return processed_trades

def x_points2(data,index):
    # index=index.tolist()[0]
    val_h=data.iloc[index]
    start_date = val_h['mtime'].date()
    t=val_h['mtime'].time()
    filtered_data = data[(data['mtime'].dt.date == start_date) & (data.index <= index)]
    # print(filtered_data)
    # print(f'filtered_data:\n{filtered_data.head()}')
    max_value = filtered_data['Close'].max()
    min_value = filtered_data['Close'].min()
    return max_value, min_value
    
    
class TradeStatus(Enum):
    BUY = 1
    SELL = -1
    NONE = 0
def distance_checker(price,value,lim):
    plh=price-value
    plh=abs(plh)
    if plh >lim:
        return 1
    else:
        return 0

def find_status(data,index,window,size):
    cspreads=[]
    # print(index,window,size)
    # print("fstatus here1")
    for i in range(window):
        datapoint=data.iloc[index-i]
        close=datapoint['Close']
        openv=datapoint['Open']
        cspread=abs(close-openv)
        cspreads.append(cspread)
    # print("fstatus here2")
    if max(cspreads)>size:
        return 1
    else:
        return 0
    
    
def daily3(data,  adjtime, lim, window, c_size, sl_adj, multiplier):
    trades = []
    trade = TradeStatus.NONE
    opendate = None
    openprice = 0
    tp = 0
    sl = 0

    for i, val_h in data.iterrows():
        time = val_h['mtime'].time()
        adj_time = pd.to_datetime(adjtime).time()

        if time < adj_time:
            continue

        tday = val_h['mtime'].date()
        target_datetime = pd.to_datetime(f'{tday} {adjtime}')
        val = data[data['mtime'] == target_datetime]
        
        startindex = val.index
        ind = startindex.tolist()[0]
        begin_max_value, begin_min_value = x_points2(data, ind)
        maxv, minv = begin_max_value, begin_min_value

        price = val_h['Close']
        dist_check = distance_checker(price, maxv if trade == TradeStatus.SELL else minv, lim)

        if trade == TradeStatus.NONE:
            if dist_check == 1 and find_status(data, i, window, c_size) == 1:
                if price > maxv:
                    trade = TradeStatus.SELL
                elif price < minv:
                    trade = TradeStatus.BUY

                openprice = price
                opendate = val_h['time']
                sl = (val_h['High'] if trade == TradeStatus.BUY else val_h['Low']) - sl_adj
                expected_loss = abs(sl - price)
                target_tp = expected_loss * multiplier
                tp = price + target_tp if trade == TradeStatus.SELL else price - target_tp

        elif trade in [TradeStatus.BUY, TradeStatus.SELL]:
            if (price > tp and trade == TradeStatus.BUY) or (price < sl and trade == TradeStatus.SELL):
                closedate = val_h['time']
                profit = calc_prof(price, openprice) if trade == TradeStatus.BUY else calc_prof(openprice, price)
                trades.append(('Buy' if trade == TradeStatus.BUY else 'Sell', opendate, closedate, profit))
                trade = TradeStatus.NONE

    return trades

def daily4(data, adjtime, lim, window, c_size, sl_adj, multiplier):
    trades = []
    trade = TradeStatus.NONE
    opendate = None
    openprice = 0
    tp = 0
    sl = 0

    adj_time = pd.to_datetime(adjtime).time()
    data['mtime'] = pd.to_datetime(data['mtime'])
    
    mask = data['mtime'].dt.time >= adj_time
    data = data[mask]

    for i, val_h in data.iterrows():
        tday = val_h['mtime'].date()
        target_datetime = pd.to_datetime(f'{tday} {adjtime}')
        val = data[data['mtime'] == target_datetime]
        
        startindex = val.index
        ind = startindex.tolist()[0]
        begin_max_value, begin_min_value = x_points2(data, ind)
        maxv, minv = begin_max_value, begin_min_value

        price = val_h['Close']
        dist_check = distance_checker(price, maxv if trade == TradeStatus.SELL else minv, lim)

        if trade == TradeStatus.NONE:
            if dist_check == 1 and find_status(data, i, window, c_size) == 1:
                if price > maxv:
                    trade = TradeStatus.SELL
                elif price < minv:
                    trade = TradeStatus.BUY

                openprice = price
                opendate = val_h['time']
                sl = (val_h['High'] if trade == TradeStatus.BUY else val_h['Low']) - sl_adj
                expected_loss = abs(sl - price)
                target_tp = expected_loss * multiplier
                tp = price + target_tp if trade == TradeStatus.SELL else price - target_tp

        elif trade in [TradeStatus.BUY, TradeStatus.SELL]:
            if (price > tp and trade == TradeStatus.BUY) or (price < sl and trade == TradeStatus.SELL):
                closedate = val_h['time']
                profit = calc_prof(price, openprice) if trade == TradeStatus.BUY else calc_prof(openprice, price)
                trades.append(('Buy' if trade == TradeStatus.BUY else 'Sell', opendate, closedate, profit))
                trade = TradeStatus.NONE

    return trades

def daily_move(data,combinations,output_csv):
    result_list=[]
    for values in combinations:
        adjtime, lim, window, c_size, sl_adj, multiplier = values
        name=f'starttime:{adjtime,} window: {window}candlesize{c_size}sl_adj{sl_adj}multiplier{multiplier}'
        result = daily3(data, adjtime, lim, window, c_size, sl_adj, multiplier)
        evaluated_info=evaluate(result,name)
        result_dict1 = {
        'Strategy': name,
        'Take_profit_level':"level1(tp1)",
        'Max_Profit': evaluated_info[0],
        'Num_Buy_Trades': evaluated_info[1],
        'Total_Buy_Profit': evaluated_info[2],
        'Num_Sell_Trades': evaluated_info[3],
        'Total_Sell_Profit': evaluated_info[4],
        'Max_Profit_Index': evaluated_info[5],
        'Index_of_Max_Profit': evaluated_info[6],
        'Max_Duration': evaluated_info[7],
        'ID_Max_Duration': evaluated_info[8],
        'total_profitable_trades':evaluated_info[9],
        'total_unprofitable_trades':evaluated_info[10],
        'total_profitable_trades':evaluated_info[11],
        'total_loss_trades':evaluated_info[12]}

        result_list.append(result_dict1)
        
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = result_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in result_list:
            writer.writerow(row)
def daily(data,adjtime,lim,window,c_size,sl_adj,multiplier):
    trades=[]
    trade=0
    opendate=None
    openprice=0
    tp=0
    sl=0
    for i in range(len(data)):
        val_h=data.loc[i]
        time=val_h['mtime'].time()
        adj_time = pd.to_datetime(adjtime).time()# adtime being such astring '03:00:00'
        if time<adj_time:
            continue
        else:
            
            tday=val_h['mtime'].date()
            target_datetime = pd.to_datetime(f'{tday} {adj_time}')
            # print(target_datetime)
            val=data[data['mtime'] == target_datetime]
            # print(val)
            # print(val.index)
            
            startindex=val.index
            ind=startindex.tolist()[0]
            begin_max_value,begin_min_value=x_points2(data,ind)
            maxv=begin_max_value
            # print(maxv)
            minv=begin_min_value
            # print(minv)
            
            # print(f'tade{trade}')
            price=val_h['Close']
            if trade==0:
                if price>maxv:
                    if distance_checker(price,maxv,lim)==0:
                        maxv=price
                        continue
                    elif distance_checker(price,maxv,lim)==1:
                        # print('i am here 4')
                        if find_status(data,i,window,c_size)==1:
                            # print('HEREEE')
                            # print(ind)
                            trade=-1
                            openprice=price
                            opendate=val_h['time']
                            lowp=val_h['Low']
                            # print('i am here ')
                            sl=lowp-sl_adj
                            expectedloss=price-sl
                            target_tp=expectedloss*multiplier
                            tp=price+target_tp
                            # print(trade)

                            
                elif price<minv:
                    if distance_checker(price,minv,lim)==0:
                        minv=price
                        continue
                    elif distance_checker(price,minv,lim)==1:
                        # print('i am here3 ')
                        if find_status(data,i,window,c_size)==1:
                            # print('found me 1')
                            trade=1
                            openprice=price
                            # print(trade)
                            opendate=val_h['time']
                            # print('i am here2 ')
                            highp=val_h['High']
                            sl=highp-sl_adj
                            expectedloss=sl-price
                            target_tp=expectedloss*multiplier
                            tp=price-target_tp
            elif trade==1:
                # print('i am here8 ')
                if price>tp:
                    # print('i am here 6')
                    closedate=val_h['time'] 
                    profit=calc_prof(price,openprice)
                    trades.append(('Buy',opendate,closedate,profit))
                    
                    trade=0
                elif price<sl:
                    # print('i am here7 ')
                    closedate=val_h['time']
                    profit=calc_prof(price,openprice)
                    trades.append(('Buy',opendate,closedate,profit))
                    
                    trade=0
            elif trade==-1:
                # print('i am here 9 ')
                if price<tp:
                    # print('i am here 10 ')
                    closedate=val_h['time']
                    profit=calc_prof(openprice,price)
                    trades.append(('Sell',opendate,closedate,profit))
                    
                    trade=0
                elif price>sl:
                    # print('i am here 11 ')
                    closedate=val_h['time']
                    profit=calc_prof(openprice,price)
                    trades.append(('Sell',opendate,closedate,profit))
                    
                    trade=0
    return trades
                    
                           
def daily_move2(data,adjtime, lim, window, c_size, sl_adj, multiplier ):
    name=f'starttime:{adjtime,} window: {window}candlesize{c_size}sl_adj{sl_adj}multiplier{multiplier}'
    result = daily(data, adjtime, lim, window, c_size, sl_adj, multiplier)
    print('i am here')
    evaluated_info=evaluate(result,name)
    print('i am here2')
    result_dict1 = {
    'Strategy': name,
    'Take_profit_level':"level1(tp1)",
    'Max_Profit': evaluated_info[0],
    'Num_Buy_Trades': evaluated_info[1],
    'Total_Buy_Profit': evaluated_info[2],
    'Num_Sell_Trades': evaluated_info[3],
    'Total_Sell_Profit': evaluated_info[4],
    'Max_Profit_Index': evaluated_info[5],
    'Index_of_Max_Profit': evaluated_info[6],
    'Max_Duration': evaluated_info[7],
    'ID_Max_Duration': evaluated_info[8],
    'total_profitable_trades':evaluated_info[9],
    'total_unprofitable_trades':evaluated_info[10],
    'total_profitable_trades':evaluated_info[11],
    'total_loss_trades':evaluated_info[12]}

    return result_dict1
    # with open(name, 'w', newline='') as csvfile:
    #     fieldnames = result_dict1.keys()
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     writer.writeheader()
    #     writer.writerow(result_dict1)
            