import pandas as pd
import numpy as np
from numba import njit, prange

@njit
def calculate_signals(open_prices, high_prices, low_prices, close_prices, upper_envelope, lower_envelope):
    n = len(close_prices)
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

    for i in prange(1, n):
        # Engulfing Bullish candle + Low outside or equal to envelope curve
        if (prev_open[i-1] < curr_close[i] and curr_open[i] < curr_close[i] and prev_open[i-1] > prev_close[i-1] and
            curr_low[i] <= lower_envelope[i]):
            engulfing_bull[i] = 1

        # Engulfing Bearish candle + High outside or equal envelope curve
        if (prev_open[i-1] > curr_close[i] and curr_open[i] > curr_close[i] and prev_open[i-1] < prev_close[i-1] and
            curr_high[i] >= upper_envelope[i]):
            engulfing_bear[i] = 1

        # Custom type of bullish hammer + Low outside or equal to envelope curve
        if (curr_open[i] < curr_close[i] and
            prev_open[i-1] < prev_close[i-1] and
            (curr_open[i] - curr_low[i]) > (0.9 * (prev_close[i-1] - low_prices[i])) and
            curr_low[i] <= lower_envelope[i]):
            hammer_bull[i] = 1

        # Custom type of bearish hammer + High outside or equal envelope curve
        if (curr_open[i] > curr_close[i] and
            prev_open[i-1] > prev_close[i-1] and
            (curr_high[i] - curr_open[i]) > (0.9 * (high_prices[i] - prev_close[i-1])) and
            curr_high[i] >= upper_envelope[i]):
            hammer_bear[i] = 1

    # Signals
    bearish_signal = engulfing_bear | hammer_bear
    bullish_signal = engulfing_bull | hammer_bull
    
    # Shift signals to avoid look-ahead bias
    bearish_signal = np.roll(bearish_signal, 1)
    bullish_signal = np.roll(bullish_signal, 1)
    bearish_signal[0] = 0
    bullish_signal[0] = 0
    
    return bearish_signal, bullish_signal

@njit
def calculate_atr(high, low, close):
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr = np.empty_like(high)
    atr[0] = np.nan
    atr[1:15] = np.mean(tr[:14])
    for i in range(15, len(atr)):
        atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    return atr

@njit
def update_trade_status(open_price, high, low, close, atr, bearish_signal, bullish_signal, 
                        prev_direction, prev_in_trade, prev_atr_trail_sl, atr_multiplier):
    n = len(open_price)
    direction = np.zeros(n, dtype=np.int32)
    in_trade = np.zeros(n, dtype=np.bool_)
    atr_trail_sl = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    entry_price = np.full(n, np.nan)
    
    direction[0] = prev_direction
    in_trade[0] = prev_in_trade
    atr_trail_sl[0] = prev_atr_trail_sl

    pending_exit = False
    pending_exit_price = np.nan

    for i in range(1, n):
        direction[i] = direction[i-1]
        in_trade[i] = in_trade[i-1]
        atr_trail_sl[i] = atr_trail_sl[i-1]

        if pending_exit:
            exit_price[i] = pending_exit_price
            direction[i] = 0
            in_trade[i] = False
            atr_trail_sl[i] = np.nan
            pending_exit = False
            pending_exit_price = np.nan
            continue

        if bullish_signal[i-1] and direction[i] != 1:
            if in_trade[i]:
                exit_price[i] = open_price[i]
            direction[i] = 1
            in_trade[i] = True
            entry_price[i] = open_price[i]
            atr_trail_sl[i] = open_price[i] - atr_multiplier * atr[i]
        elif bearish_signal[i-1] and direction[i] != -1:
            if in_trade[i]:
                exit_price[i] = open_price[i]
            direction[i] = -1
            in_trade[i] = True
            entry_price[i] = open_price[i]
            atr_trail_sl[i] = open_price[i] + atr_multiplier * atr[i]

        if in_trade[i]:
            if direction[i] == 1 and low[i] <= atr_trail_sl[i]:
                pending_exit = True
                pending_exit_price = max(low[i], atr_trail_sl[i])
                if entry_price[i] == open_price[i]:
                    entry_price[i] = np.nan
            elif direction[i] == -1 and high[i] >= atr_trail_sl[i]:
                pending_exit = True
                pending_exit_price = min(high[i], atr_trail_sl[i])
                if entry_price[i] == open_price[i]:
                    entry_price[i] = np.nan

        if in_trade[i] and not pending_exit:
            if direction[i] == 1:
                new_sl = close[i] - atr_multiplier * atr[i]
                atr_trail_sl[i] = max(new_sl, atr_trail_sl[i])
            else:
                new_sl = close[i] + atr_multiplier * atr[i]
                atr_trail_sl[i] = min(new_sl, atr_trail_sl[i])

    return direction, in_trade, atr_trail_sl, exit_price, entry_price

@njit
def calculate_envelopes(ewm_short, ewm_long, envelopes_perc):
    upper_envelope = np.empty_like(ewm_short)
    lower_envelope = np.empty_like(ewm_short)
    
    for i in range(len(ewm_short)):
        if ewm_short[i] > ewm_long[i]:
            upper_envelope[i] = ewm_short[i] * (1 + envelopes_perc)
            lower_envelope[i] = ewm_short[i] * (1 - envelopes_perc / 2)
        else:
            upper_envelope[i] = ewm_short[i] * (1 + envelopes_perc / 2)
            lower_envelope[i] = ewm_short[i] * (1 - envelopes_perc)
    
    return upper_envelope, lower_envelope

@njit
def process_dataframe_numba(open_values, high_values, low_values, close_values, atr_multiplier, ewm_short, ewm_long, envelopes_perc):
    n = len(close_values)
    
    atr = calculate_atr(high_values, low_values, close_values)
    
    upper_envelope, lower_envelope = calculate_envelopes(ewm_short, ewm_long, envelopes_perc)
    
    bearish_signal, bullish_signal = calculate_signals(
        open_values, high_values, low_values, close_values, upper_envelope, lower_envelope
    )
    
    direction, in_trade, atr_trail_sl, exit_price, entry_price = update_trade_status(
        open_values, high_values, low_values, close_values, atr,
        bearish_signal, bullish_signal,
        0, False, np.nan, atr_multiplier
    )
    
    return atr, direction, in_trade, atr_trail_sl, exit_price, entry_price, bearish_signal, bullish_signal, upper_envelope, lower_envelope

def process_dataframe(df, atr_multiplier=5, ewm_period=50, envelopes_perc=0.01):
    # Extract numpy arrays from DataFrame
    open_values = df['Open'].values
    high_values = df['High'].values
    low_values = df['Low'].values
    close_values = df['Close'].values
    
    # Calculate EWM and envelopes
    ewm_short = df['Close'].ewm(span=ewm_period, adjust=False).mean().values
    ewm_long = df['Close'].ewm(span=ewm_period*2, adjust=False).mean().values
    
    # Process data using Numba-optimized function
    atr, direction, in_trade, atr_trail_sl, exit_price, entry_price, bearish_signal, bullish_signal, upper_envelope, lower_envelope = process_dataframe_numba(
        open_values, high_values, low_values, close_values, atr_multiplier, ewm_short, ewm_long, envelopes_perc
    )
    
    # Assign results back to DataFrame
    df['ATR'] = atr
    df['direction'] = direction
    df['in_trade'] = in_trade
    df['ATR_trail_sl'] = atr_trail_sl
    df['entry_price'] = entry_price
    df['exit_price'] = exit_price
    df['bearish_signal'] = bearish_signal
    df['bullish_signal'] = bullish_signal
    df[f'EWM_{ewm_period}'] = ewm_short
    df[f'EWM_{ewm_period*2}'] = ewm_long
    df['Upper_Envelope'] = upper_envelope
    df['Lower_Envelope'] = lower_envelope
    
    return df