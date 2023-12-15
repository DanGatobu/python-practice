from bolinger import bolinger_stategy
from functions import evaluate
import itertools

def run():
    rsi_values = [6, 8, 12,14,16,24,32]
    sma_values = [10, 8, 6,12 ]
    standard_deviations = [2,3,4,7]
    window_values = [6,4,8,10,14]

    for rsi_value, sma_value, standard_deviation, window_value in itertools.product(rsi_values, sma_values, standard_deviations, window_values):
        trades, data = bolinger_stategy(rsi_value, sma_value, standard_deviation, window_value)
        evaluate(trades, f"bolinger sma{sma_value} rsi{rsi_value} window{window_value} standard deviation {standard_deviation}")
    
if __name__=='__main__':
    run()
    
    