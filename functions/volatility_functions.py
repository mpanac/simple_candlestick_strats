import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

def calculate_ewma_volatility(data, 
                            timeframe='15min',
                            decay_factor=None,
                            min_periods=30):
    """
    Calculate EWMA (Exponentially Weighted Moving Average) Volatility.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        OHLCV data with DatetimeIndex
    timeframe : str, default '15min'
        Data frequency ('1min', '3min', '5min', '10min', '15min', '20min', '30min', '45min', '1h', etc.)
    decay_factor : float, optional
        Custom decay factor between 0 and 1. If None, will be automatically selected
        based on timeframe
    min_periods : int, default 30
        Minimum number of observations required to calculate volatility
        
    Returns:
    --------
    pandas.Series
        EWMA volatility series
    """
    # Validate inputs
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")
    
    if 'Close' not in data.columns:
        raise ValueError("Data must contain 'Close' column")
    
    # Set default decay factors based on timeframe
    default_decay_factors = {
        '1min':  0.90,
        '3min':  0.905,
        '5min':  0.91,
        '10min': 0.915,
        '15min': 0.92,
        '20min': 0.925,
        '30min': 0.93,
        '45min': 0.935,
        '1h':    0.94,
        '2h':    0.95,
        '4h':    0.96,
        '12h':   0.97,
        '1d':    0.94,
    }
    
    # If no decay factor provided, use default based on timeframe
    if decay_factor is None:
        decay_factor = default_decay_factors.get(timeframe, 0.94)  # Default to 0.94 if timeframe not found
    
    # Validate decay factor
    if not 0 < decay_factor < 1:
        raise ValueError("Decay factor must be between 0 and 1")
    
    # Calculate returns
    returns = np.log(data['Close'] / data['Close'].shift(1))
    
    # Calculate squared returns
    squared_returns = returns ** 2
    
    # Calculate EWMA variance - using only alpha parameter
    ewma_variance = squared_returns.ewm(
        alpha=1-decay_factor,
        min_periods=min_periods,
        adjust=False
    ).mean()
    
    # Convert variance to volatility (annualized)
    timeframe_to_annual = {
        '1min':  525600,    # 365 * 24 * 60
        '3min':  175200,    # 365 * 24 * 20
        '5min':  105120,    # 365 * 24 * 12
        '10min': 52560,     # 365 * 24 * 6
        '15min': 35040,     # 365 * 24 * 4
        '20min': 26280,     # 365 * 24 * 3
        '30min': 17520,     # 365 * 24 * 2
        '45min': 11680,     # 365 * 24 * 4/3
        '1h':    8760,      # 365 * 24
        '2h':    4380,      # 365 * 12
        '4h':    2190,      # 365 * 6
        '12h':   730,       # 365 * 2
        '1d':    365,
    }
    
    annualization_factor = timeframe_to_annual.get(timeframe, 365)  # Default to 365 if timeframe not found
    volatility = np.sqrt(ewma_variance * annualization_factor) * 100  # Convert to percentage
    
    return volatility

def add_volatility_bands(volatility, num_stdev=2, window=30):
    """
    Add volatility bands to help identify high/low volatility regimes using rolling statistics.
    
    Parameters:
    -----------
    volatility : pandas.Series
        Calculated volatility series
    num_stdev : float, default 2
        Number of standard deviations for bands
    window : int, default 30
        Rolling window size for calculating mean and standard deviation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with volatility and regime indicators containing:
        - volatility: original volatility values
        - volatility_regime: 'high', 'normal', or 'low'
        - vol_upper_band: upper volatility band
        - vol_lower_band: lower volatility band
    """
    # Calculate rolling volatility statistics
    vol_mean = volatility.rolling(window=window).mean()
    vol_std = volatility.rolling(window=window).std()
    
    # Create bands
    upper_band = vol_mean + (vol_std * num_stdev)
    lower_band = vol_mean - (vol_std * num_stdev)
    
    # Create regime indicator
    vol_regime = pd.Series(index=volatility.index, dtype='str')
    vol_regime[volatility > upper_band] = 'high'
    vol_regime[volatility < lower_band] = 'low'
    vol_regime[(volatility >= lower_band) & (volatility <= upper_band)] = 'normal'
    
    # Combine into results
    results = pd.DataFrame({
        'volatility': volatility,
        'volatility_regime': vol_regime,
        'vol_upper_band': upper_band,
        'vol_lower_band': lower_band,
        'vol_mean': vol_mean  # Added for reference
    })
    
    return results




def plot_price_and_volatility(price_data, volatility_data, window_days=180):
    """
    Create a visualization of price and volatility with regime bands.
    
    Parameters:
    -----------
    price_data : pandas.DataFrame
        OHLCV data with DatetimeIndex
    volatility_data : pandas.DataFrame
        Volatility analysis data with volatility, regime, and bands
    window_days : int, default 180
        Number of days to display in the plot
    """
    # Calculate the start date based on window_days
    end_date = price_data.index[-1]
    start_date = end_date - pd.Timedelta(days=window_days)
    
    # Filter data for the selected window
    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    price_window = price_data[mask]
    vol_window = volatility_data[mask]
    
    # Create figure and grid
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Price subplot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(price_window.index, price_window['Close'], label='Price', color='black', linewidth=1)
    ax1.set_title('Price and Volatility Analysis', pad=20)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Color the background based on volatility regime
    min_price = price_window['Close'].min()
    max_price = price_window['Close'].max()
    price_range = max_price - min_price
    
    for idx in range(len(vol_window)-1):
        if vol_window['volatility_regime'].iloc[idx] == 'high':
            color = 'red'
            alpha = 0.1
        elif vol_window['volatility_regime'].iloc[idx] == 'low':
            color = 'green'
            alpha = 0.1
        else:
            continue  # Skip normal regime
            
        ax1.axvspan(vol_window.index[idx], 
                    vol_window.index[idx+1], 
                    ymin=0, ymax=1,
                    color=color, alpha=alpha)
    
    # Volatility subplot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(vol_window.index, vol_window['volatility'], 
             label='Volatility', color='blue', linewidth=1)
    ax2.plot(vol_window.index, vol_window['vol_upper_band'], 
             label='Upper Band', color='red', linestyle='--', alpha=0.7)
    ax2.plot(vol_window.index, vol_window['vol_lower_band'], 
             label='Lower Band', color='green', linestyle='--', alpha=0.7)
    ax2.plot(vol_window.index, vol_window['vol_mean'], 
             label='Mean', color='gray', linestyle='-', alpha=0.7)
    
    ax2.set_ylabel('Volatility (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig


# Basic usage
# volatility = calculate_ewma_volatility(
#     data=raw_ohlcv,
#     timeframe='15min',  # Your data frequency
#     window=30  # Number of periods to consider
# )

# # With custom decay factor
# volatility = calculate_ewma_volatility(
#     data=raw_ohlcv,
#     timeframe='15min',
#     decay_factor=0.92,  # Custom decay
#     window=30
# )

# # Get volatility with regime bands
# vol_analysis = add_volatility_bands(
#     volatility=volatility,
#     num_stdev=2
# )

# # Example of using it for your exit strategy
# def determine_markout_period(vol_regime):
#     if vol_regime == 'high':
#         return 30  # 30 minutes for high volatility
#     elif vol_regime == 'low':
#         return 90  # 90 minutes for low volatility
#     else:
#         return 60  # 60 minutes for normal volatility