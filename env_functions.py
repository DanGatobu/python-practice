def candle_info(open,close,high=0,low=0):
        candle_size = 0
        candle_spread = 0
        candle_type = {}

        if open > close:
            candle_size=open-close
            candle_spread=high-low
            candle_type={'candletype':'red','spread':candle_spread,'size':candle_size}
            
        elif open < close:
            candle_size=close-open
            candle_spread=low-high
            candle_type={'candletype':'green','spread':candle_spread,'size':candle_size}
        return candle_type
    
def multiplier(size):
    if size < 2.3:
        return 1
    elif size >= 2.3 and size < 4.6:
        return 2
        
def candle_info(open,close,high=0,low=0):
    candle_size = 0
    candle_spread = 0
    candle_type = {}

    if open > close:
        candle_size=open-close
        candle_spread=high-low
        candle_type={'candletype':'red','spread':candle_spread,'size':candle_size}
        
    elif open < close:
        candle_size=close-open
        candle_spread=high-low
        candle_type={'candletype':'green','spread':candle_spread,'size':candle_size}
    return candle_type