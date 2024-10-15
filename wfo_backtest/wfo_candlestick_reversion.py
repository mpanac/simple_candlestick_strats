import vectorbt as vbt
from itertools import product
from tqdm import tqdm
import multiprocessing
import os
import importlib
import logging
import sys
from functools import partial
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory of 'wfo_backtest' to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from functions.custom_functions import wfo_rolling_split_params
from strats.candlestick_reversion import process_dataframe


def calculate_signals(bullish_signal, bearish_signal, direction, exit_price):
    
    bullish_signal = bullish_signal.astype(bool)
    bearish_signal = bearish_signal.astype(bool)
    
    long_entries = (bullish_signal.shift(1) & (direction.shift(1) != 1))
    short_entries = (bearish_signal.shift(1) & (direction.shift(1) != -1))
    short_exits = (~np.isnan(exit_price) & (direction.shift(1) == -1))
    long_exits = (~np.isnan(exit_price) & (direction.shift(1) == 1))
    
    return long_entries, short_entries, short_exits, long_exits

def process_param_combination(df, params, freq, fees, init_cash):
    param_dict = dict(zip(['window', 'desired_return', 'atr_multiplier'], params))
    
    df = process_dataframe(df.copy(), **param_dict)
    
    long_entries, short_entries, short_exits, long_exits = calculate_signals(
        df['bullish_signal'], df['bearish_signal'], 
        df['direction'], df['exit_price']
    )
    
    open_prices = df['Open'].where(np.isnan(df['exit_price']), df['exit_price'])
    
    pf = vbt.Portfolio.from_signals(
        open_prices,
        entries=long_entries,
        short_entries=short_entries,
        exits=long_exits,
        short_exits=short_exits,
        freq=freq,
        fees=fees,
        init_cash=init_cash,
        accumulate=False
    )
    
    stats = pf.stats()
    
    return {
        'params': param_dict,
        'sharpe_ratio': stats['Sharpe Ratio'],
        'sortino_ratio': stats['Sortino Ratio'],
        'calmar_ratio': stats['Calmar Ratio'],
        'total_return': stats['Total Return [%]'] / 100,
        'max_drawdown': stats['Max Drawdown [%]'] / 100,
        'total_trades': stats['Total Trades'],
        'stats': stats
    }

## Use this to select the one with most trades among top 5
# def auto_select_params(results_df):
#     # Select top 5 results based on Sharpe ratio
#     top_5 = results_df.head(5)
    
#     # Choose the strategy with the most trades among the top 5
#     selected = top_5.loc[top_5['total_trades'].idxmax()]
    
#     return selected['params']

def auto_select_params(results_df):
    # Select the top 1 result (first row) based on Sharpe ratio
    selected = results_df.iloc[0]
    
    return selected['params']


def process_window(i, in_ohlcv_i, out_ohlcv_i, param_ranges, freq, fees, init_cash, auto_select, subfolder):
    param_combinations = list(product(*param_ranges.values()))
    
    with multiprocessing.Pool() as pool:
        process_func = partial(process_param_combination, in_ohlcv_i, freq=freq, fees=fees, init_cash=init_cash)
        results = list(tqdm(pool.imap(process_func, param_combinations), 
                            total=len(param_combinations), 
                            desc=f"Processing window {i+1}"))
    
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['total_trades'] > 0]  # Filter out results with no trades
    results_df = results_df.sort_values(['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return'], 
                                        ascending=[False, False, False, False]).reset_index(drop=True)
    
    plot_filename = create_combined_parameter_sharpe_plot(results, i, subfolder)
    
    logger.info(f"Combined Parameter-Sharpe plot saved as {plot_filename}")
    
    if auto_select:
        chosen_params = auto_select_params(results_df)
        logger.info(f"Automatically selected parameters: {chosen_params}")
    else:
        logger.info("Please review the plot and choose parameters for the out-of-sample test.")
        # User input for parameter selection
        chosen_params = {}
        for param, values in param_ranges.items():
            while True:
                try:
                    value = float(input(f"Enter value for {param}: "))
                    if value in values:
                        chosen_params[param] = value
                        break
                    else:
                        print(f"Value not in range. Please choose from: {values}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
    
    # Process out-of-sample data with chosen parameters
    out_result = process_param_combination(out_ohlcv_i, tuple(chosen_params.values()), freq, fees, init_cash)
    
    return {
        'window': i,
        'in_sample_results': results_df.head(5).to_dict('records'),
        'chosen_params': chosen_params,
        'out_sample_result': out_result,
        'plot_filename': plot_filename
    }

def save_results_to_csv(results, output_file):
    data = []
    for window_result in results:
        window_data = {
            'Window': window_result['window'] + 1,
            **window_result['chosen_params'],
            **{f'OutOfSample_{k}': v for k, v in window_result['out_sample_result']['stats'].items()}
        }
        data.append(window_data)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to CSV: {output_file}")

def save_results(results, output_file_md, output_file_csv):
    # Save to markdown
    with open(output_file_md, 'w') as f:
        f.write("# Walk-Forward Optimization Results\n\n")
        for window_result in results:
            f.write(f"## Window {window_result['window'] + 1}\n\n")
            
            f.write("### Chosen Parameters for Out-of-Sample Test\n")
            for param, value in window_result['chosen_params'].items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
            
            f.write("### Out-of-Sample Results\n")
            out_result = window_result['out_sample_result']
            stats = out_result['stats']
            
            f.write(f"Start: {stats['Start']:}\n")
            f.write(f"End: {stats['End']:}\n")
            f.write(f"Total Return: {stats['Total Return [%]']:.2f}%\n")
            f.write(f"Benchmark Return: {stats['Benchmark Return [%]']:.2f}%\n")
            f.write(f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%\n")
            f.write(f"Sharpe Ratio: {stats['Sharpe Ratio']:.4f}\n")
            f.write(f"Sortino Ratio: {stats['Sortino Ratio']:.4f}\n")
            f.write(f"Calmar Ratio: {stats['Calmar Ratio']:.4f}\n")
            f.write(f"Win Rate: {stats['Win Rate [%]']:.2f}%\n")
            f.write(f"Total Trades: {stats['Total Trades']}\n")
            f.write("\n")

    logger.info(f"Results saved to markdown: {output_file_md}")

    # Save to CSV
    save_results_to_csv(results, output_file_csv)

def walk_forward_optimization(in_ohlcv, out_ohlcv, in_indexes, out_indexes, param_ranges, freq, fees, init_cash, auto_select, subfolder):
    results = []
    for i in range(len(in_indexes)):
        result = process_window(i, in_ohlcv[i], out_ohlcv[i], param_ranges, freq, fees, init_cash, auto_select, subfolder)
        results.append(result)
        
        logger.info(f"\nWindow {i+1} Results:")
        logger.info(f"In-sample top Sharpe: {result['in_sample_results'][0]['sharpe_ratio']:.4f}")
        logger.info(f"Chosen parameters: {result['chosen_params']}")
        logger.info(f"Out-of-sample Sharpe: {result['out_sample_result']['sharpe_ratio']:.4f}")
    
    return results

def create_combined_parameter_sharpe_plot(results, window_index, subfolder):
    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    df = df.sort_values('sharpe_ratio', ascending=False)
    
    # Create a parameter combination string for each result
    df['param_combination'] = df['params'].apply(lambda x: ', '.join(f"{k}:{v}" for k, v in x.items()))
    
    # Calculate Calmar ratio
    df['calmar_ratio'] = df['total_return'] / df['max_drawdown'].abs()
    
    # Create the scatter plot
    fig = go.Figure()
    
    hover_text = df.apply(lambda row: f"Params: {row['param_combination']}<br>"
                                      f"Sharpe Ratio: {row['sharpe_ratio']:.4f}<br>"
                                      f"Sortino Ratio: {row['sortino_ratio']:.4f}<br>"
                                      f"Calmar Ratio: {row['calmar_ratio']:.4f}<br>"
                                      f"Total Return: {row['total_return']:.2%}<br>"
                                      f"Total Trades: {row['total_trades']}", axis=1)
    
    scatter = go.Scatter(
        x=list(range(len(df))),
        y=df['sharpe_ratio'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['sharpe_ratio'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Sharpe Ratio')
        ),
        text=hover_text,
        hoverinfo='text'
    )
    
    fig.add_trace(scatter)
    
    # Add horizontal lines at Sharpe ratios -2, 0, and 2
    for y in [-2, 0, 2]:
        fig.add_shape(
            type="line",
            x0=0,
            y0=y,
            x1=len(df) - 1,
            y1=y,
            line=dict(color="red" if y != 0 else "green", width=2, dash="dash"),
        )
    
    # Update layout
    fig.update_layout(
        title=f'Parameter Combinations vs Sharpe Ratio (Window {window_index + 1})',
        xaxis_title='Parameter Combinations (sorted by Sharpe Ratio)',
        yaxis_title='Sharpe Ratio',
        hovermode='closest',
        height=800,  # Increased height
        width=1200,  # Increased width
    )
    
    # Update x-axis
    fig.update_xaxes(showticklabels=False)  # Hide x-axis labels as they're now in the hover text
    
    # Ensure the directory exists
    os.makedirs(subfolder, exist_ok=True)
    
    # Save the plot as an interactive HTML file
    plot_filename = os.path.join(subfolder, f'combined_parameter_sharpe_plot_window_{window_index}.html')
    fig.write_html(plot_filename)
    
    return plot_filename

if __name__ == "__main__":
    # Extract the strategy name from the import
    strategy_module = importlib.import_module('strats.candlestick_reversion')
    strategy_name = strategy_module.__name__.split('.')[-1]

    input_file = './data/binance_data/btcusdt_ohlcv_30m.csv'  # Change the data path to use different granularity. DON'T FORGET TO ALSO CHANGE FREQ variable !!!
    pair_freq = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create a subfolder structure that includes the strategy name
    subfolder = os.path.join('results', strategy_name, pair_freq)
    os.makedirs(subfolder, exist_ok=True)


    raw_ohlcv = pd.read_csv(input_file, parse_dates=['Date'], index_col='Date')
    logger.info(f"Loaded data with {len(raw_ohlcv)} rows")

    # You can tweak the insample_percentage as well as the n number of periods for walk-forward optimization. This will output new variables which i process in my custom function 'wfo_rolling_split_params'
    n, window_len, set_lens = wfo_rolling_split_params(total_candles=len(raw_ohlcv), insample_percentage=0.80, n=16)
    logger.info(f"Data split into {n} windows")
    
    # This function will split our data making them ready to perform walk-forward backtest
    (in_ohlcv, in_indexes), (out_ohlcv, out_indexes) = raw_ohlcv.vbt.rolling_split(
        n=n,
        window_len=window_len,
        set_lens=set_lens,
    )

    # Change the param ranges based on your preferences 
    param_ranges = {
        'window': range(15, 71, 5),
        'desired_return': [round(x, 3) for x in np.arange(0.005, 0.0251, 0.0025)],
        'atr_multiplier': range(1, 9)
    }
    
    
    freq = '30m' # Binance taker fee 0.05%, maker fee 0.025%
    fees = 0.0005  # Binance taker fee 0.05%, maker fee 0.025%
    init_cash = 100000

     # This allows you either select your own parameters combination each processed window or let automatically select the params combination from Top 5 of the last in-sample test
    while True:
        user_input = input("Do you want to automatically select parameters? (y/n): ").lower()
        if user_input in ['y', 'n']:
            auto_select = (user_input == 'y')
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    results = walk_forward_optimization(in_ohlcv, out_ohlcv, in_indexes, out_indexes, param_ranges, freq, fees, init_cash, auto_select, subfolder)

    # Save results in the subfolder
    save_results(results, 
                 os.path.join(subfolder, 'wfo_results.md'), 
                 os.path.join(subfolder, 'wfo_results.csv'))