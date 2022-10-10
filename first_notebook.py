"""
Perry J. Kaufman - Alpha Trading_ Profitable Strategies That Remove Directional Risk (Wiley Trading)  -Wiley (2011).pdf
"""
import matplotlib.pyplot as plt
from lib_glob.constants import *
import numpy as np
import pandas as pd
import seaborn
import sys
import statsmodels
from statsmodels.tsa.stattools import coint

import numpy as np
import pandas as pd

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
# just set the seed for the random number generator
np.random.seed(107)

import matplotlib.pyplot as plt

# just set the seed for the random number generator
np.random.seed(107)

test_start = "2015-01-01"
test_end = "2016-01-01"
stock_list = pd.DataFrame()
PORTFOLIO = BIST30
# PORTFOLIO.append('XU100')

for stock in PORTFOLIO:
    df = pd.read_csv(path_stock_daily + stock + ".csv", index_col="date", usecols=["date", "close"])
    df = df.loc[(df.index > test_start) & (df.index <= test_end)]
    stock_list[stock] = df['close']
    # if stock == 'XU100':
    #     df.index = pd.to_datetime(df.index)
    #     df = df.resample('H').ffill()
    #     stock_list[stock] = df['close']
    #     print(stock_list[stock])
    #     sys.exit()

stock_list.replace([np.inf, -np.inf], np.nan, inplace=True)
stock_list.fillna(method='ffill', inplace=True)
stock_list.fillna(method='bfill', inplace=True)


def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            # print(keys[i], keys[j])
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.02:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs



scores, pvalues, pairs = find_cointegrated_pairs(stock_list)
seaborn.heatmap(pvalues, xticklabels=PORTFOLIO, yticklabels=PORTFOLIO, cmap='RdYlGn_r'
                , mask=(pvalues >= 0.02)
                )

print(len(pairs), pairs)
plt.show()
#
# stock_list['AKBNK'].plot()
# stock_list['TKFEN'].plot()
# plt.show()

"""
Calculating the Spread
Now we will plot the spread of the two series. In order to actually calculate the spread,
 we use a linear regression to get the coefficient for the linear combination to construct between our two securities, 
 as shown in the stationarity lecture. Using a linear regression to estimate the coefficient
is known as the Engle-Granger method.

"""
S1 = stock_list['AKBNK']
S2 = stock_list['TKFEN']

S1 = sm.add_constant(S1)
results = sm.OLS(S2, S1).fit()
S1 = S1['AKBNK']
b = results.params['AKBNK']

spread = S2 - b * S1
# spread.plot()
# plt.axhline(spread.mean(), color='black')
# plt.legend(['Spread'])
# plt.show()

"""
Alternatively, we could examine the ratio betwen the two series.

ratio = S1/S2
ratio.plot()
plt.axhline(ratio.mean(), color='black')
plt.legend(['Price Ratio']);


Examining the price ratio of a trading pair is a traditional way to handle pairs trading. Part of why this works as a signal is based in our assumptions of how stock prices move, specifically because stock prices are typically assumed to be log-normally distributed. What this implies is that by taking a ratio of the prices, we are taking a linear combination of the returns associated with them (since prices are just the exponentiated returns).
This can be a little irritating to deal with for our purposes as purchasing the precisely correct ratio of a trading pair may not be practical. We choose instead to move forward with simply calculating the spread between the cointegrated stocks using linear regression. This is a very simple way to handle the relationship, however, and is likely not feasible for non-toy examples. There are other potential methods for estimating the spread listed at the bottom of this lecture. If you want to get more into the theory of why having cointegrated stocks matters for pairs trading, again, please see the Integration, Cointegration, and Stationarity Lecture from the Quantopian Lecture Series.
So, back to our example. The absolute spread isn't very useful in statistical terms. It is more helpful to normalize our signal by treating it as a z-score.


In practice this is usually done to try to give some scale to the data, but this assumes some underlying distribution, 
usually a normal distribution. Under a normal distribution, we would know that approximately 84% of all spread values will be smaller. 
However, much financial data is not normally distributed, and one must be very careful not to assume normality,
 nor any specific distribution when generating statistics. It could be the case that the true distribution of spreads was
  very fat-tailed and prone to extreme values. This could mess up our model and result in large losses.

"""

def zscore(series):
    return (series - series.mean()) / np.std(series)

# zscore(spread).plot()
# plt.axhline(zscore(spread).mean(), color='black')
# plt.axhline(1.0, color='red', linestyle='--')
# plt.axhline(-1.0, color='green', linestyle='--')
# plt.legend(['Spread z-score', 'Mean', '+1', '-1'])
# plt.show()

"""
Simple Strategy:
Go "Long" the spread whenever the z-score is below -1.0
Go "Short" the spread when the z-score is above 1.0
Exit positions when the z-score approaches zero
This is just the tip of the iceberg, and only a very simplistic example to illustrate the concepts. In practice you would want to compute a more optimal weighting for how many shares to hold for S1 and S2. Some additional resources on pair trading are listed at the end of this notebook

Trading using constantly updating statistics
In general taking a statistic over your whole sample size can be bad. For example, if the market is moving up, and both securities with it, then your average price over the last 3 years may not be representative of today. For this reason traders often use statistics that rely on rolling windows of the most recent data.

Moving Averages
A moving average is just an average over the last  n  datapoints for each given time. It will be undefined for the first  n  datapoints in our series. Shorter moving averages will be more jumpy and less reliable, but respond to new information quickly. Longer moving averages will be smoother, but take more time to incorporate new information.

We also need to use a rolling beta, a rolling estimate of how our spread should be calculated, in order to keep all of our parameters up to date.

"""

# Get the spread between the 2 stocks
# Calculate rolling beta coefficient
rolling_beta = sm.OLS(y=S1, x=S2, window_type='rolling', window=30)
spread = S2 - rolling_beta.beta['x'] * S1
spread.name = 'spread'

# Get the 1 day moving average of the price spread
spread_mavg1 = pd.rolling_mean(spread, window=1)
spread_mavg1.name = 'spread 1d mavg'

# Get the 30 day moving average
spread_mavg30 = pd.rolling_mean(spread, window=30)
spread_mavg30.name = 'spread 30d mavg'

plt.plot(spread_mavg1.index, spread_mavg1.values)
plt.plot(spread_mavg30.index, spread_mavg30.values)

plt.legend(['1 Day Spread MAVG', '30 Day Spread MAVG'])

plt.ylabel('Spread');

"""
We can use the moving averages to compute the z-score of the spread at each given time. This will tell us how extreme 
the spread is and whether it's a good idea to enter a position at this time. Let's take a look at the z-score now
"""
# Take a rolling 30 day standard deviation
std_30 = pd.rolling_std(spread, window=30)
std_30.name = 'std 30d'

# Compute the z score for each day
zscore_30_1 = (spread_mavg1 - spread_mavg30)/std_30
zscore_30_1.name = 'z-score'
zscore_30_1.plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--');

"""
The z-score doesn't mean much out of context, let's plot it next to the prices to get an idea of what it looks like. 
We'll take the negative of the z-score because the spreads were all negative and that is a little counterintuitive to trade on.
"""

# Plot the prices scaled down along with the negative z-score
# just divide the stock prices by 10 to make viewing it on the plot easier
plt.plot(S1.index, S1.values/10)
plt.plot(S2.index, S2.values/10)
plt.plot(zscore_30_1.index, zscore_30_1.values)
plt.legend(['S1 Price / 10', 'S2 Price / 10', 'Price Spread Rolling z-Score']);

"""
Out of Sample Test
Now that we have constructed our spread appropriately and have an idea of how we will go about making trades, it is time to conduct some out of sample testing. Our whole model is based on the premise that these securities are cointegrated, but we built it on information from a certain time period. If we actually want to implement this model, we need to conduct an out of sample test to confirm that the principles of our model are still valid going forward.
Since we initially built the model on the 2014 - 2015 year, let's see if this cointegrated relationship holds for 2015 - 2016. Historical results do not guarantee future results so this is a sanity check to see if the work we have done holds strong.
"""

symbol_list = ['ABGB', 'FSLR']
prices_df = get_pricing(symbol_list, fields=['price']
                               , start_date='2015-01-01', end_date='2016-01-01')['price']
prices_df.columns = map(lambda x: x.symbol, prices_df.columns)
S1 = prices_df['ABGB']
S2 = prices_df['FSLR']
score, pvalue, _ = coint(S1, S2)
print('p-value: ', pvalue)

"""
Unfortunately, since our p-value is above the cutoff of  0.05 , we conclude that our model will no longer be valid due to the lack of cointegration between our chosen securities. If we tried to deploy this model without the underlying assumptions holding, we would have no reason to believe that it would actually work. Out of sample testing is a vital step to make sure that our work will actually be viable in the market.

Implementation
When actually implementing a pairs trading strategy you would normally want to be trading many different pairs at once. If you find a good pair relationship by analyzing data, there is no guarantee that that relationship will continue into the future. Trading many different pairs creates a diversified portfolio to mitigate the risk of individual pairs "falling out of" cointegration.
There is a template algorithm attached to this lecture that shows an example of how you would implement pairs trading on our platform. Feel free to check it out and modify it with your own pairs to see if you can improve it.

Further Research
This notebook contained some simple introductory approaches. In practice one should use more sophisticated statistics, some of which are listed here.

Augmented-Dickey Fuller test
Hurst exponent
Half-life of mean reversion inferred from an Ornstein–Uhlenbeck process
Kalman filters
(this is not an endorsement) But, a very good practical resource for learning more about pair trading is Dr. Ernie Chan's book: Algorithmic Trading: Winning Strategies and Their Rationale

"""

