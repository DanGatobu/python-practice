from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import MetaTrader5 as mt5
import plotly.graph_objects as go
import numpy as np
from backtesting import Backtest
from backtesting import Strategy
def isgreen(close,op):
    isgree=None
    if close>op:
        isgree=True
    return isgree
def isred(close,op):
    isre=None
    if close<op:
        isre=True
    return isre
def issellpattern(data):
    for i in range(3, len(data)):
        row = data.iloc[i]
        prev1 = data.iloc[(i - 1)]
        prev2 = data.iloc[(i - 2)]
        prev3 = data.iloc[(i - 3)]
        
        prev1_close = prev1['Close']
        prev1_open = prev1['Open']
        prev1_status = isgreen(prev1_close, prev1_open)
        
        case2prev1_status = isred(prev1_close, prev1_open)
        
        prev2_open = prev2['Open']
        prev2_close = prev2['Close']
        prev3_close = prev3['Open']
        prev3_open = prev3['Close']
        
        prev2_status = isred(prev2_close, prev2_open)
        case2prev2 = isgreen(prev2_close, prev2_open)
        prev3_status = isred(prev3_close, prev3_open)
        
        nowo = row['Open']
        nowc = row['Close']
        now_status = isred(nowc, nowo)
        
        issl = 0
        if now_status and prev1_status and prev2_status:
            issl = 1
        data.at[i, 'sell'] = issl  
        issl=0
        
    return data

def isbuypattern(data):
    for i in range(3, len(data)):
        row = data.iloc[i]
        prev1 = data.iloc[(i - 1)]
        prev2 = data.iloc[(i - 2)]
        prev3 = data.iloc[(i - 3)]
        
        prev1_close = prev1['Close']
        prev1_open = prev1['Open']
        prev1_status = isred(prev1_close, prev1_open)
        
        case2prev1_status = isgreen(prev1_close, prev1_open)
        
        prev2_open = prev2['Open']
        prev2_close = prev2['Close']
        prev3_close = prev3['Open']
        prev3_open = prev3['Close']
        
        prev2_status = isgreen(prev2_close, prev2_open)
        case2prev2 = isred(prev2_close, prev2_open)
        prev3_status = isgreen(prev3_close, prev3_open)
        
        nowo = row['Open']
        nowc = row['Close']
        now_status = isgreen(nowc, nowo)
        
        isbyp = 0
        if now_status and prev1_status and prev2_status:
            isbyp = 1
        data.at[i, 'buypattern'] = isbyp
        isbyp=0
        
    return data
def resistance_point(data, l, r):
    isres = 0
    for i in range(5, len(data) - 5):
        row = data.iloc[i]
        inprev = True
        innxt = True
        prev = data.iloc[i - l : i]
        for _, v in prev.iterrows():
            if row['High'] < v['High']:
                inprev = False
                break

        nxt = data.iloc[i + 1 : i + r + 1]
        for _, v in nxt.iterrows():
            if row['High'] < v['High']:
                innxt = False
                break

        if inprev and innxt:
            isres = 1
        data.at[i, 'Resistance'] = isres
        isres = 0

    return data

def supportfunc(data, l, r):
    isres = 0
    for i in range(l, len(data) - r):
        row = data.iloc[i]
        inprev = True
        innxt = True
        prev = data.iloc[i - l : i]
        for _, v in prev.iterrows():
            if row['Low'] > v['Low']:
                inprev = False
                break

        nxt = data.iloc[i + 1 : i + r + 1]
        for _, v in nxt.iterrows():
            if row['Low'] > v['Low']:
                innxt = False
                break

        if inprev and innxt:
            isres = 1
        data.at[i, 'Support'] = isres
        isres = 0

    return data

mt5.initialize()
ohlc_data= pd.DataFrame (mt5.copy_rates_range("GOLD",mt5.TIMEFRAME_M15,datetime(2021, 1, 1),datetime.now()))
labels = ['small', 'medium', 'mediumhigh', 'high']

# Create the 'volume_class' column based on the 'tick_volume' column
ohlc_data['volume_class'] = pd.qcut(ohlc_data['tick_volume'], q=4, labels=labels)
coll=['spread','real_volume']
ohlc_data.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'},inplace=True)
ohlc_data.drop(columns=coll,axis=1,inplace=True)
ohlc_data=ohlc_data.dropna()
ohlc_data.drop(ohlc_data.loc[ohlc_data['tick_volume'] == 0].index, inplace=True)


teach = np.array_split(ohlc_data, 3)
teach1=teach[0]
teach1=issellpattern(teach1)
teach1 = supportfunc(teach1, 5, 5)
teach1=isbuypattern(teach1)
teach1=resistance_point(teach1, 5, 5)
holder=teach1[:400]

resistance=[]
recent_resistance = None
for i in range(len(holder)):
    
    isbyp=0
    if holder.iloc[i]['Resistance']==1:
        resistance.append(i)
        recent_resistance = holder.iloc[i]['High']
    if recent_resistance is not None and holder.iloc[i]['High']>recent_resistance and holder.iloc[i]['buypattern']==1:
         isbyp=1
    holder.at[i, 'buy'] = isbyp
    isbyp=0
        
for i in range(len(holder)):
    if holder.iloc[i]['buy']==1:
        sl =holder.iloc[i]['Low']-5.0
        nowprice=holder.iloc[i]['Close']
        value=abs(nowprice-sl)*2
        tp=nowprice+value
        holder.at[i, 'sl'] = sl
        holder.at[i, 'tp'] = tp
sup = []
sells = []

recent_sup = None

for i in range(len(holder)):
    iss = 0
    if holder.iloc[i]['Support'] == 1:
        sup.append(i)
        recent_sup = holder.iloc[i]['Low']
    if recent_sup is not None and holder.iloc[i]['Low'] < recent_sup and holder.iloc[i]['sell'] == 1:
        iss = 1
    holder.at[i, 'sells'] = iss
    iss = 0
for i in range(len(holder)):
    if holder.iloc[i]['sells']==1:
        sl =holder.iloc[i]['High']+5.0
        nowprice=holder.iloc[i]['Close']
        value=abs(nowprice-sl)*2
        tp=nowprice-value
        holder.at[i, 'ssl'] = sl
        holder.at[i, 'stp'] = tp
class breakout(Strategy):

    def init(self):
        pass
           
    def next(self):

        if  self.data.buy==1:
#             print(self.data.Close)
            sl=self.data.sl
            tp=self.data.tp
            self.buy(size=0.1,sl=sl,tp=tp)
        elif self.data.sells==1:
#             print(self.data.ssl)
            ssl=self.data.ssl
            stp=self.data.stp
            self.sell(size=0.1,sl=ssl,tp=stp)
            

        
holder=holder.set_index("time")
holder.index.name = None
holder.index = pd.to_datetime(holder.index, unit='s').strftime('%Y-%m-%d %H:%M:%S')
holder.index = pd.to_datetime(holder.index)

bt = Backtest(holder, breakout, cash=1000000, commission=.002,exclusive_orders=True,trade_on_close=True)


stats = bt.run()

print(stats)
bt.plot()