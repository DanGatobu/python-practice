from functions import get_data,bolinger_function,rsi,calculate_sma,calc_prof
import numpy as np
def bolinger_stategy(rsi_value,sma_value,standard_dev_value,window_value):
    data=get_data("GOLD")
    l_rsi=[]
    l_rsi.append(rsi_value)
    data=rsi(data,l_rsi)
    l_sma=[]
    l_sma.append(sma_value)
    data=calculate_sma(data,l_sma)
    data=bolinger_function(data,std=standard_dev_value,tpw=window_value,s_num=sma_value)
    conditions=[(data.rsi_6 < 30) & (data.Close < data.lower_band),
    (data[f'rsi_{rsi_value}'] > 70) & (data.Close > data.upper_band)]
    choices = ['Buy', 'Sell']
    data ['signal'] = np.select(conditions, choices)
    data.signal=data.signal.shift()
    data['shifted']=data.Close.shift()
    trades = []

    bposition = False
    sposition = False
    buyprices = 0
    startdate = 0
    enddate = 0
    sellprice = 0
    sell_enddate = 0
    sell_startdate = 0
    bbprice = []

    for index, row in data.iterrows():
        if not bposition and row['signal'] == 'Buy':
            if sposition==True:
                exitprice = row.Open
                sell_enddate = row.time
                profit = calc_prof(sellprice, exitprice)
                sposition = False
                trades.append(('Sell',startdate, sell_enddate, profit))
            bbprice.append(row.Open)
            buyprices = row.Open
            startdate = row.time
            bposition = True

        if bposition:
            if row['signal'] == 'Sell' or row.shifted < 0.95 * bbprice[-1]:
                sellprice = row.Open
                sell_startdate = row.time
                exitprice = row.Open
                enddate = row.time
                sposition = True
                profit = calc_prof(sellprice, buyprices)
                bposition = False
                trades.append(('Buy',startdate, enddate, profit))

    return trades,data


