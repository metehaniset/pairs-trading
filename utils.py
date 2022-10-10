import numpy as np
import pandas as pd
import statsmodels
import talib
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import warnings
import pickle
from statsmodels.regression.rolling import RollingOLS
from tqdm import tqdm


def get_data(portfolio=BIST30, start_date='2005-01-01', end_date='2010-01-01', period='G'):
    stock_list = pd.DataFrame()
    path = path_stock + period + '/'
    for stock in portfolio:
        df = pd.read_csv(path + stock + ".csv", index_col="date", parse_dates=True, usecols=["date", "close"])
        data_start_index = df.head(1).index[0]
        # df.index = pd.to_datetime(df.index)
        df = df.loc[(df.index > start_date) & (df.index <= end_date)]
        # df = pd.DataFrame(df, index=pd.date_range(start_date, end_date, freq="B"))
        # df.index.name = 'date'

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # df[df.index < data_start_index] = df[df.index < data_start_index].fillna(0)
        # df.dropna(inplace=True)

        stock_list[stock] = df['close']

    stock_list.replace([np.inf, -np.inf], np.nan, inplace=True)
    stock_list.fillna(method='ffill', inplace=True)
    stock_list.dropna(axis=1, inplace=True)
    # stock_list.fillna(method='bfill', inplace=True)
    # stock_list.fillna(0, inplace=True)
    stock_list.dropna(axis=0, inplace=True)

    return stock_list


def find_cointegrated_pairs(data, max_pvalue=0.02, check_stationary=True):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    pairs_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            try:
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                if not is_stationary(S1, cutoff=0.10) and check_stationary:
                    continue
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = coint(S1, S2)
                    if len(w):
                        # if w[0].category == statsmodels.tools.sm_exceptions.CollinearityWarning:
                            # print(keys[i], keys[j], 'not including in pairs',  w[0].message)
                        # print(keys[i] + '-' + keys[j], i, j, n,  w[0].message)
                        continue

                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                # print(keys[i] + '-' + keys[j], i, j, pvalue)
                if pvalue < max_pvalue:
                    # pairs.append((keys[i], keys[j]))
                    pairs_dict[keys[i] + '-' + keys[j]] = pvalue
            except:
                # print(keys[i], keys[j], 'contains NaN')
                continue

    combined_sorted = sorted(pairs_dict.items(), key=lambda x: x[1], reverse=False)
    # print(combined_sorted)
    for i in range(len(pairs_dict)):
        stock1, stock2 = combined_sorted[i][0].split('-')
        pairs.append((stock1, stock2))
    return score_matrix, pvalue_matrix, pairs


def find_all_pairs(df, rolling_type='rolling', rolling_window='120 days', resample_window='W-FRI',
                   check_stationary=True, data_period='G'):
    pairs_dict = {}
    recalc_dates = df.resample(resample_window).mean().index.values[:-1]
    print('Preparing pairs for ', resample_window, 'data_period:', data_period)
    for index in recalc_dates:   # tqdm(recalc_dates):
        if rolling_type == 'rolling':
            data = df[(df.index > (index - pd.Timedelta(rolling_window))) & (df.index <= index)]
        elif rolling_type == 'expanding':
            data = df[df.index <= index]

        if len(data) < 20:
            # print(index, 'Not calculating because of len(data) < 20')
            continue

        # print(data.head(10).to_string())
        # sys.exit()
        scores, pvalues, pairs = find_cointegrated_pairs(data, check_stationary=check_stationary)
        # print(index)
        if len(pairs) == 0:
            print(index, 0, [])
            pairs_dict[pd.to_datetime(str(index)).strftime('%Y-%m-%d')] = []
            continue
        # else:
        pairs_dict[pd.to_datetime(str(index)).strftime('%Y-%m-%d')] = pairs
        print(index, len(pairs), pairs)

    print(pairs_dict)
    pair_df = pd.DataFrame.from_dict(pairs_dict, orient='index').fillna(0)
    # print(pair_df)
    pair_df.index = pd.to_datetime(pair_df.index)

    if data_period == 'G':
        pair_df = pair_df.resample('B').ffill()
    elif data_period == '60':
        pair_df = pair_df.resample('H').ffill()

    pair_df = pair_df.reindex(df.index).ffill()
    pair_df.replace(0, np.nan, inplace=True)
    print(pair_df)

    return pair_df


def is_stationary(closes, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(closes)[1]
    if pvalue < cutoff:
        return True
        # print('p-value = ' + str(pvalue) + ' The series ' + stock.name +' is likely stationary.')
    else:
        return False
        # print('p-value = ' + str(pvalue) + ' The series ' + stock.name +' is likely non-stationary.')


def calculate_ols_zscore_ratio(df, ols_type='expanding', ols_window=120):
    portfolio_len = len(df.columns)
    portfolio = list(df.columns)
    zscore_df = pd.DataFrame()
    for i in range(portfolio_len):
        for j in range(portfolio_len):
            if i == j:  # Aynı hisseyi aynı hisseyle pair etme
                continue

            stock1 = portfolio[i]
            stock2 = portfolio[j]
            S1 = df[stock1]
            S2 = df[stock2]
            S1 = sm.add_constant(S1)

            # results = sm.OLS(S2, S1).fit()    # Bu overfit eder
            if ols_type == 'rolling':
                results = RollingOLS(S2, S1, window=ols_window, min_nobs=20).fit()   # Rolling OLS
                S1 = S1[stock1]
                b = results.params[stock1]
                spread = S2 - b * S1
            elif ols_type == 'expanding':
                results = sm.RecursiveLS(S2, S1).fit()  # Expanding OLS
                b = results.recursive_coefficients.filtered[0]
                S1 = S1[stock1]
                spread = S2 - b * S1

            ratios_ma1 = spread.rolling(window=1, center=False).mean()
            ratios_ma2 = spread.rolling(window=20, center=False).mean()
            rolling_std = spread.rolling(window=20, center=False).std()
            zscore = (ratios_ma1 - ratios_ma2) / rolling_std
            zscore_df[stock1 + '-' + stock2] = zscore

    return zscore_df


def calculate_sub_ratio(df, zscore_window=20):
    portfolio_len = len(df.columns)
    portfolio = list(df.columns)
    zscore_df = pd.DataFrame()
    for i in range(portfolio_len):
        for j in range(portfolio_len):
            if i == j:  # Aynı hisseyi aynı hisseyle pair etme
                continue

            stock1 = portfolio[i]
            stock2 = portfolio[j]
            S1 = df[stock1]
            S2 = df[stock2]
            spread = S2 - S1

            ratios_ma1 = spread.rolling(window=1, center=False).mean()
            ratios_ma2 = spread.rolling(window=zscore_window, center=False).mean()
            rolling_std = spread.rolling(window=zscore_window, center=False).std()
            zscore = (ratios_ma1 - ratios_ma2) / rolling_std
            zscore_df[stock1 + '-' + stock2] = zscore

    return zscore_df.dropna()


def calculate_momentum_ratio(df, method='rsi', zscore_window=20):
    portfolio_len = len(df.columns)
    portfolio = list(df.columns)
    result_df = pd.DataFrame()
    for i in range(portfolio_len):
        for j in range(portfolio_len):
            if i == j:  # Aynı hisseyi aynı hisseyle pair etme
                continue

            stock1 = portfolio[i]
            stock2 = portfolio[j]
            S1 = df[stock1]
            S2 = df[stock2]
            if method == 'rsi':
                mom_s1, _ = talib.STOCHRSI(S1, timeperiod=14)
                mom_s2, _ = talib.STOCHRSI(S2, timeperiod=14)
            spread = mom_s2 - mom_s1

            result_df[stock1 + '-' + stock2] = spread
            # ratios_ma1 = spread.rolling(window=1, center=False).mean()
            # ratios_ma2 = spread.rolling(window=zscore_window, center=False).mean()
            # rolling_std = spread.rolling(window=zscore_window, center=False).std()
            # zscore = (ratios_ma1 - ratios_ma2) / rolling_std
            # result_df[stock1 + '-' + stock2] = zscore

    return result_df.dropna()



def get_pairs_at_index(pairs_df, index):
    pairs_list = []
    for pairs in list(pairs_df.loc[index].dropna()):
        # print(index, pairs)
        # for pair in pairs:
        stock1, stock2 = pairs[0], pairs[1]
        pairs_list.append(stock1 + '-' + stock2)
    # print(index, pairs_list)
    return pairs_list
