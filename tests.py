from pairs_trading.lib.utils import *
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

start_date = '2005-01-01'
end_date = '2006-01-01'
df = get_data(portfolio=BIST30, start_date=start_date, end_date=end_date)
# for c in df.columns:
#     stationarity_test(df[c])
# sys.exit()

scores, pvalues, pairs = find_cointegrated_pairs(df)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(pvalues, xticklabels=BIST30, yticklabels=BIST30, cmap='RdYlGn_r', mask=(pvalues >= 0.05))
plt.show()
print(pairs)
# print(pvalues)


stock1 = pairs[1][0]
stock2 = pairs[1][1]
S1 = df[stock1]
S2 = df[stock2]
"""
Now we can plot the spread of the two time series. 
In order to actually calculate the spread, we use a linear regression to get the coefficient for the linear combination 
to construct between our two securities, as mentioned with the Engle-Granger method before.

"""
S1 = sm.add_constant(S1)
results = sm.OLS(S2, S1).fit()
S1 = S1[stock1]
b = results.params[stock1]

spread = S2 - b * S1
spread.plot(figsize=(12, 6))
plt.axhline(spread.mean(), color='black')
# plt.xlim(start_date, end_date)
plt.legend(['Spread'])
plt.show()
ratio = spread

"""
Alternatively, we can examine the ration between the two time series

"""
# ratio = S1/S2
# print(ratio.to_string())
#
# ratio.plot(figsize=(12, 6))
# plt.axhline(ratio.mean(), color='black')
# # plt.xlim(start_date, end_date)
# plt.legend(['Price Ratio'])
# plt.show()

# zscore(ratio).plot(figsize=(12, 6))
# plt.axhline(zscore(ratio).mean())
# plt.axhline(1.0, color='red')
# plt.axhline(-1.0, color='green')
# # plt.xlim(start_date, end_date)
# plt.show()


ratios_mavg5 = ratio.rolling(window=1, center=False).mean()
ratios_mavg60 = ratio.rolling(window=20, center=False).mean()
std_60 = ratio.rolling(window=20, center=False).std()
zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
zscore_60_5.plot(figsize=(12, 6))
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(ratio.index, ratio.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)
plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])

plt.ylabel('Ratio')
plt.show()



