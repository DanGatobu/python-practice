import concurrent.futures

from functions import joined,get_data,calculate_sma
using_data=get_data('GOLD')



using_data=calculate_sma(using_data,[3,7,5,8,10,18,21,25,28,30,15,33,38,12,40,35])
list_sma=[3,7,5,8,10,18,21,25,28,30,15,33,38,12,40,35]
window_list=[5,10,15,20,25,30,35,40,45,50]
smatrials=list_sma
window=window_list
combinations = [(small_sma, large_sma, win) for small_sma in smatrials for large_sma in smatrials if small_sma != large_sma and small_sma < large_sma for win in window]

# Example usage
if __name__ == "__main__":
    
    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use map to apply the joined function to each combination in parallel
        results_list = list(
            executor.map(joined, [using_data] * len(combinations), *zip(*combinations))
        )

    # Now you can access the results_list, each element contains the results for one combination
    for result_set in results_list:
        result_dict1, result_dict2, result_dict3, dfs = result_set
        break
        # Process or use the results as needed...
