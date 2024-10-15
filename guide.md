run: pip install -r requirements.txt

1. In the strats folder i have the strategies where i pass the OHLCV data. These data gets processed and returns adjusted dataframe with signals included. Then it is used in the wfo_backtest

2. To test the strategies run the python scripts in wfo_backtest/ folder. There are different scripts and each script matches the strategy based on naming.

3. To visualize the strategies go to /visualization/test_strats.ipynb and play with it.

4. To see parameters combinations from each in-sample optimization period, open the .html file from the /results folder in your browser and you can see all of the tested combinations.

### I think 30minute dataframe works best with the 'candlestick_reversion.py' strategy. There are periods where sharpe ratios go around 1.0 - 2.5 on out of sample data. Also periods with sharpe negative tho. ###