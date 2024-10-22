import pandas as pd
import numpy as np
from numba import njit, prange

@njit
def calculate_log_return(close):
    return np.log(close[1:] / close[:-1])

@njit
def calculate_rolling_sum(arr, window):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        if i < window:
            result[i] = np.sum(arr[:i+1])
        else:
            result[i] = np.sum(arr[i-window+1:i+1])
    return result

@njit
def calculate_percentage_candle_size(high, low):
    return np.abs((high - low) / low) * 100

@njit
def calculate_rolling_mean(arr, window):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        if i < window:
            result[i] = np.mean(arr[:i+1])
        else:
            result[i] = np.mean(arr[i-window+1:i+1])
    return result

@njit(parallel=True)
def calculate_signals(log_return, window, open_prices, high_prices, low_prices, close_prices, desired_return, avg_candle_size, curr_candle_size):
    n = len(log_return)
    period_return = calculate_rolling_sum(log_return, window)

    prev_open = open_prices[:-1]
    prev_close = close_prices[:-1]
    curr_open = open_prices[1:]
    curr_close = close_prices[1:]
    curr_high = high_prices[1:]
    curr_low = low_prices[1:]

    engulfing_bull = np.zeros(n, dtype=np.int32)
    engulfing_bear = np.zeros(n, dtype=np.int32)
    hammer_bull = np.zeros(n, dtype=np.int32)
    hammer_bear = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        # Engulfing Bullish pattern
        if prev_open[i] < curr_close[i] and curr_open[i] < curr_close[i] and prev_open[i] > prev_close[i]:
            engulfing_bull[i] = 1

        # Engulfing Bearish pattern
        if prev_open[i] > curr_close[i] and curr_open[i] > curr_close[i] and prev_open[i] < prev_close[i]:
            engulfing_bear[i] = 1

        # Hammer Bullish pattern
        if (curr_open[i] < curr_close[i] and
            prev_open[i] < prev_close[i] and
            (curr_open[i] - curr_low[i]) > (0.9 * (prev_close[i] - low_prices[i]))):
            hammer_bull[i] = 1

        # Hammer Bearish pattern
        if (curr_open[i] > curr_close[i] and
            prev_open[i] > prev_close[i] and
            (curr_high[i] - curr_open[i]) > (0.9 * (high_prices[i] - prev_close[i]))):
            hammer_bear[i] = 1

    bearish_signal = ((engulfing_bear | hammer_bear) &
                      (curr_candle_size > avg_candle_size * 1.5) &
                      (period_return >= desired_return))
    
    bullish_signal = ((engulfing_bull | hammer_bull) &
                      (curr_candle_size > avg_candle_size * 1.5) &
                      (period_return <= -desired_return))
    
    return bearish_signal, bullish_signal

@njit
def update_trade_status(open_price, high, low, close, bearish_signal, bullish_signal, 
                        prev_direction, prev_in_trade, prev_fixed_sl):
    n = len(open_price)
    direction = np.zeros(n, dtype=np.int32)
    in_trade = np.zeros(n, dtype=np.bool_)
    fixed_sl = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    entry_price = np.full(n, np.nan)
    
    direction[0] = prev_direction
    in_trade[0] = prev_in_trade
    fixed_sl[0] = prev_fixed_sl

    pending_exit = False
    pending_exit_price = np.nan

    for i in range(1, n):
        direction[i] = direction[i-1]
        in_trade[i] = in_trade[i-1]
        fixed_sl[i] = fixed_sl[i-1]

        if pending_exit:
            exit_price[i] = pending_exit_price
            direction[i] = 0
            in_trade[i] = False
            fixed_sl[i] = np.nan
            pending_exit = False
            pending_exit_price = np.nan
            continue

        if bullish_signal[i-1] and direction[i] != 1:
            if in_trade[i]:
                exit_price[i] = open_price[i]
            direction[i] = 1
            in_trade[i] = True
            entry_price[i] = open_price[i]
            fixed_sl[i] = low[i-1] - 1  # Set SL $1 below the low of the signal candle
        elif bearish_signal[i-1] and direction[i] != -1:
            if in_trade[i]:
                exit_price[i] = open_price[i]
            direction[i] = -1
            in_trade[i] = True
            entry_price[i] = open_price[i]
            fixed_sl[i] = high[i-1] + 1  # Set SL $1 above the high of the signal candle

        if in_trade[i]:
            if direction[i] == 1 and low[i] <= fixed_sl[i]:
                pending_exit = True
                pending_exit_price = max(low[i], fixed_sl[i])
                if entry_price[i] == open_price[i]:
                    entry_price[i] = np.nan
            elif direction[i] == -1 and high[i] >= fixed_sl[i]:
                pending_exit = True
                pending_exit_price = min(high[i], fixed_sl[i])
                if entry_price[i] == open_price[i]:
                    entry_price[i] = np.nan

    return direction, in_trade, fixed_sl, exit_price, entry_price

@njit
def process_dataframe_numba(open_values, high_values, low_values, close_values, window, desired_return):
    n = len(close_values)
    log_return = np.empty(n)
    log_return[0] = np.nan
    log_return[1:] = calculate_log_return(close_values)
    
    percentage_candle_size = calculate_percentage_candle_size(high_values, low_values)
    avg_candle_size = calculate_rolling_mean(percentage_candle_size, window)
    
    bearish_signal, bullish_signal = calculate_signals(
        log_return[1:], window, 
        open_values[:-1], high_values[:-1], low_values[:-1], close_values[:-1],
        desired_return, avg_candle_size[:-1], percentage_candle_size[:-1]
    )
    
    direction, in_trade, fixed_sl, exit_price, entry_price = update_trade_status(
        open_values, high_values, low_values, close_values,
        bearish_signal, bullish_signal,
        0, False, np.nan
    )
    
    # Pre-allocate arrays with correct size and fill them directly
    final_direction = np.empty(n, dtype=np.int32)
    final_direction[0] = 0
    final_direction[1:] = direction[:-1]

    final_in_trade = np.empty(n, dtype=np.bool_)
    final_in_trade[0] = False
    final_in_trade[1:] = in_trade[:-1]

    final_fixed_sl = np.empty(n)
    final_fixed_sl[0] = np.nan
    final_fixed_sl[1:] = fixed_sl[:-1]

    final_entry_price = np.empty(n)
    final_entry_price[0] = np.nan
    final_entry_price[1:] = entry_price[:-1]

    final_bearish_signal = np.empty(n, dtype=np.bool_)
    final_bearish_signal[0] = False
    final_bearish_signal[1:] = bearish_signal

    final_bullish_signal = np.empty(n, dtype=np.bool_)
    final_bullish_signal[0] = False
    final_bullish_signal[1:] = bullish_signal
    
    return log_return, percentage_candle_size, avg_candle_size, final_direction, final_in_trade, final_fixed_sl, exit_price, final_entry_price, final_bearish_signal, final_bullish_signal

def process_dataframe(df, window=30, desired_return=0.01):
    # Extract numpy arrays from DataFrame
    open_values = df['Open'].values
    high_values = df['High'].values
    low_values = df['Low'].values
    close_values = df['Close'].values
    
    # Process data using Numba-optimized function
    log_return, percentage_candle_size, avg_candle_size, direction, in_trade, fixed_sl, exit_price, entry_price, bearish_signal, bullish_signal = process_dataframe_numba(
        open_values, high_values, low_values, close_values, window, desired_return
    )
    
    # Assign results back to DataFrame
    df['Log_Return'] = log_return
    df['percentage_candle_size'] = percentage_candle_size
    df['avg_candle_size'] = avg_candle_size
    df['direction'] = direction
    df['in_trade'] = in_trade
    df['fixed_sl'] = fixed_sl
    df['entry_price'] = entry_price
    df['exit_price'] = exit_price
    df['bearish_signal'] = bearish_signal
    df['bullish_signal'] = bullish_signal
    
    return df