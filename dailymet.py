# print(last_row['mtime'].time())
import datetime
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import schedule
import time as tt

def fget_data(currency,num_c):
    mt5.initialize()
    ohlc_data = pd.DataFrame(mt5.copy_rates_from_pos(currency, mt5.TIMEFRAME_M15, 0,num_c))
    ohlc_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    ohlc_data.dropna(inplace=True)

    return ohlc_data




def distance_checker(price,value,lim):
    plh=price-value
    plh=abs(plh)
    if plh >lim:
        return 1
    else:
        return 0
    
def find_status(data,size):
    data['AbsDiff'] = abs(data['Close'] - data['Open'])

# Find the maximum value among the calculated differences
    max_diff = data['AbsDiff'].max()
    if max_diff>size:
        return 1
    else:
        return 0
    

maxholder=2088.31
minholder=2071.18
def get_first_values():
    global maxholder
    global minholder
    nowd=fget_data('GOLD',110)
    nowd['mtime']=pd.to_datetime(nowd['time'],unit='s')
    last_row = nowd.iloc[-2]
    target_date = last_row['mtime'].date()

    # Filter the DataFrame for the specific day
    day_data = nowd[nowd['mtime'].dt.date == target_date]

    # Find the maximum and minimum points for the day
    max_price = day_data['High'].max()
    min_price = day_data['Low'].min()
    maxholder = max_price
    minholder = min_price
    

def trade_():
    global maxholder
    global minholder
    print(maxholder)
    print(minholder)
    nowd=fget_data('GOLD',110)
    nowd['mtime']=pd.to_datetime(nowd['time'],unit='s')
    last_row = nowd.iloc[-2]
    pg=last_row['mtime'].time()
    if not datetime.time(0, 0) <= pg <= datetime.time(3, 0):
        pos_info = mt5.positions_total()

    # Check if there are open positions or active orders
        if pos_info==0:


    # Filter the DataFrame for the specific day
            

    # Find the maximum and minimum points for the day
            max_price = maxholder
            min_price = minholder
            price=last_row['Close']
        # print(price)
            if price>max_price:
                if distance_checker(price,max_price,1.5)==0:
                    maxholder=price
                elif distance_checker(price,max_price,1.5)==1:
                    win_data=nowd.tail(11)
                    if find_status(win_data,1.5):
                        lowp=last_row['Low']
                        sl=lowp-1.5
                        expectedloss=price-sl
                        target_tp=expectedloss*5
                        tp=price+target_tp
                        lot=0.1
                        deviation=1
                        request = {"action": mt5.TRADE_ACTION_DEAL,"symbol": 'GOLD',"volume": lot,"type": mt5.ORDER_TYPE_BUY,"price": price,"sl": sl,"tp": tp,"deviation": deviation,"magic": 2002,"comment": "python script open","type_time": mt5.ORDER_TIME_GTC,"type_filling": mt5.ORDER_FILLING_IOC,}
                        mt5.order_send(request)
                    
            elif price<min_price:
                if distance_checker(price,min_price,1.5)==0:
                    minholder=price
                elif distance_checker(price,min_price,1.5)==1:
                    win_data=nowd.tail(11)
                    if find_status(win_data,1.5):
                        highp=last_row['High']
                        sl=highp+1.5
                        expectedloss=sl-price
                        target_tp=expectedloss*5
                        tp=price-target_tp
                        lot=0.1
                        deviation=1
                        request = {"action": mt5.TRADE_ACTION_DEAL,"symbol": 'GOLD',"volume": lot,"type": mt5.ORDER_TYPE_SELL,"price": price,"sl": sl,"tp": tp,"deviation": deviation,"magic": 2002,"comment": "python script open","type_time": mt5.ORDER_TIME_GTC,"type_filling": mt5.ORDER_FILLING_IOC,}
                        mt5.order_send(request)
                            
                        
                
def run_schedule():
    schedule.every().day.at('03:00').do(get_first_values)
    schedule.every().hour.at(':15').do(trade_)
    schedule.every().hour.at(':30').do(trade_)
    schedule.every().hour.at(':45').do(trade_)
    schedule.every().hour.at(':00').do(trade_)

    while True:
        schedule.run_pending()
        tt.sleep(1)

if __name__ == "__main__":
    run_schedule()       

        