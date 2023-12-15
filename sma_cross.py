from functions import get_data , calculate_sma

df=get_data('GOLD')

df=calculate_sma(df,[3,20])
