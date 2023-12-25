import concurrent.futures
from functions import daily_move2,get_data
import pandas as pd
import itertools

us_data=get_data('GOLD')
us_data['mtime']=pd.to_datetime(us_data['time'],unit='s')
us_data = us_data[us_data['mtime'].dt.date != pd.to_datetime('2015-01-02').date()]
us_data.reset_index(drop=True, inplace=True)

times=['03:00:00','04:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00']
distancelim=[2.5,3.0,4.0,5.5,6.5,7.5,9.0,8.0]
candlewindowlimit=[2,1,3,4,5,6,7,8,9,10]
candlesizes=[2.5,3.0,4.0,5.5,6.5,7.5,9.0,8.0]
sl_adj=[2.5,3.0,4.0,5.5,6.5,7.5,9.0,8.0]
multiplier=[2.5,3.0,4.0,5.5,6.5,7.5,9.0,8.0]





combinations = list(itertools.product(times, distancelim, candlewindowlimit, candlesizes, sl_adj, multiplier))
print(len(combinations))

if __name__ == "__main__":
    
    # Create a ProcessPoolExecutor
    result_df=[]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use map to apply the joined function to each combination in parallel
        results_list = executor.map(daily_move2, [us_data] * len(combinations), *zip(*combinations))
        result_df.append(results_list)

    results_df = pd.DataFrame(results_list)

# Save the DataFrame to a CSV file
    results_df.to_csv('output_file.csv', index=False)

