from utils import *
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from statsmodels.tsa.stattools import coint, adfuller
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

start_date = '2005-01-01'
end_date = '2010-01-01'
df = get_data(portfolio=BIST30, start_date=start_date, end_date=end_date)
# for c in df.columns:
#     stationarity_test(df[c])
# sys.exit()



# pairs_dict = find_rolling_cointegrated_pairs(df, rolling_window='90 days', resample_window='W-FRI')
# scores, pvalues, pairs = find_cointegrated_pairs(df)

stock1 = 'AKBNK'  # pairs[0][0]
stock2 = 'TOASO'    # pairs[0][1]
print(stock1, stock2, 'using for pairs trading')
S1 = df[stock1]
S2 = df[stock2]

S1 = sm.add_constant(S1)
results = sm.OLS(S2, S1).fit()
results = RollingOLS(S2, S1, window=125, min_nobs=20).fit()
# print(results.params)
# results = sm.RecursiveLS(S2, S1).fit()
# print(results.recursive_coefficients)
# sys.exit()

S1 = S1[stock1]
b = results.params[stock1]
spread = S2 - b * S1
# spread = S2 - S1
# spread = S2 - results.recursive_coefficients.filtered[0]*S1
ratio = spread
# print(ratio)
# sys.exit()

# ratio = S1/S2
# print(ratio)
# print(spread)
# sys.exit()
# spread.plot(figsize=(12, 6))
# plt.axhline(spread.mean(), color='black')
# # plt.xlim(start_date, end_date)
# plt.legend(['Spread'])
# plt.show()

ratios_ma1 = ratio.rolling(window=1, center=False).mean()
ratios_ma2 = ratio.rolling(window=20, center=False).mean()
rolling_std = ratio.rolling(window=20, center=False).std()
zscore = (ratios_ma1 - ratios_ma2)/rolling_std

zscore.dropna(inplace=True)

capital = 10000
orders = {}
for index in zscore.index:
    if zscore[index] < -1:
        if len(orders) > 0:
            # print('len(orders) > 0 in zscore[index] < -1, not generating new orders')
            continue
        o1 = {'stock': stock1, 'index': index, 'side': 'SHORT', 'price': df[stock1][index], 'amount': int((capital / 2) / df[stock1][index])}
        o2 = {'stock': stock2, 'index': index, 'side': 'LONG', 'price': df[stock2][index], 'amount': int((capital / 2) / df[stock2][index])}
        capital -= (o1['amount'] * o1['price'] + o2['amount']*o2['price'])
        orders[stock1+'-'+stock2] = [o1, o2]
        print(index, 'zscore[index] < -1')
        # LONG s1, short s2
        pass
    elif zscore[index] > 1:
        if len(orders) > 0:
            # print('len(orders) > 0 in zscore[index] > 1, not generating new orders')
            continue
        o1 = {'stock': stock1, 'index': index, 'side': 'LONG', 'price': df[stock1][index], 'amount': int((capital / 2) / df[stock1][index])}
        o2 = {'stock': stock2, 'index': index, 'side': 'SHORT', 'price': df[stock2][index], 'amount': int((capital / 2) / df[stock2][index])}
        capital -= (o1['amount'] * o1['price'] + o2['amount']*o2['price'])
        orders[stock1+'-'+stock2] = [o1, o2]
        print(index, 'zscore[index] > 1')
        # SHORT s1, LONG s2
        pass
    elif abs(zscore[index]) < 0.90:
        if len(orders) > 0:
            for o in orders[stock1+'-'+stock2]:
                current_price = df[o['stock']][index]
                # print(index, o, current_price)
                if o['side'] == 'LONG':
                    gorl = (current_price - o['price'])*o['amount']
                    capital += (o['price']*o['amount'] + gorl)

                elif o['side'] == 'SHORT':
                    gorl = (o['price'] - current_price)*o['amount']
                    capital += (o['price']*o['amount'] + gorl)

                commision = capital * 0.0003
                capital -= commision

            del orders[stock1 + '-' + stock2]
            print(index, 'FLAT', capital)
