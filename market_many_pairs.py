from utils import *
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.rolling import RollingOLS
import sys
import pickle
import matplotlib.pyplot as plt
import pyfolio as pf
import seaborn as sns
sns.set(style="whitegrid")

PICKLE_PATH = '/home/pairs_trading/pickle/'
start_date = '2015-01-01'   # BAŞLANGIÇ TARİHİNİ DEĞİŞTİRİRSEM PAİR SAYISI DA DEĞİŞECEK NaN'Lardan DOLAYI
end_date = '2020-01-01'
PORTFOLIO = BIST30
data_period = "G"
df = get_data(portfolio=PORTFOLIO, start_date=start_date, end_date=end_date, period=data_period)
print('Using stock counts:', len(df.columns))
PORTFOLIO = list(df.columns)
# for c in df.columns:
#     stationarity_test(df[c])
# sys.exit()

rolling_type = 'rolling'
rolling_window = '90 days'     # expanding'de kullanılılmıyor bu parametre,
resample_window = 'W'
RECALCULATE = False      # Veriler baştan hesapla ya da daha önce hesapladığını kullan

if rolling_type == 'rolling':
    data_pickle_suffix = rolling_type + '_' + rolling_window.replace(' ', '') + '_' + resample_window + '_' + start_date + '_' + end_date
elif rolling_type == 'expanding':
    data_pickle_suffix = rolling_type + '_' + resample_window + '_' + start_date + '_' + end_date

print(data_pickle_suffix)
try:
    assert not RECALCULATE
    pairs_df = pickle.load(open(PICKLE_PATH + 'pairs_' + data_pickle_suffix + '.pickle', 'rb'))
except:
    pairs_df = find_all_pairs(df, rolling_type=rolling_type, rolling_window=rolling_window,
                              resample_window=resample_window, check_stationary=True, data_period=data_period)
    pickle.dump(pairs_df,  open(PICKLE_PATH + 'pairs_' + data_pickle_suffix + '.pickle', 'wb'))


# Veri seti hazırlandı
# Şimdi score kısmını hazırla, şu an OLS kullanıyorum, stockastik de ekleyeceğim, p1/p2 de ekleyeceğim
ols_type = 'rolling'
ols_window = rolling_window.split(' ')[0]
if ols_type == 'rolling':
    zscore_pickle_suffix = 'zscore_rolling_ols_' + str(ols_window) + '_'
elif ols_type == 'expanding':
    zscore_pickle_suffix = 'zscore_expanding_ols_'

# try:
#     # assert not RECALCULATE
#     zscore_df = pickle.load(open(PICKLE_PATH + zscore_pickle_suffix + data_pickle_suffix + '.pickle', 'rb'))
# except:
#     print('Calculating ols zscore ratio', zscore_pickle_suffix)
#     zscore_df = calculate_ols_zscore_ratio(df, ols_type=ols_type, ols_window=int(ols_window))
#     pickle.dump(zscore_df,  open(PICKLE_PATH + zscore_pickle_suffix + data_pickle_suffix + '.pickle', 'wb'))

zscore_df = calculate_sub_ratio(df, zscore_window=20) # Fena değil
# zscore_df = calculate_momentum_ratio(df, method='rsi',zscore_window=50)
zscore_df.dropna(inplace=True)
# print(zscore_df.head(10).to_string())
# print(list(zscore_df.columns))
# print(len(zscore_df))
#
# print(pairs_df)


def flat(pair_name, o, reason='unknown_reason'):
    global capital, number_of_transactions
    current_price = df[o['stock']][index]
    # print(index, o, current_price)
    if o['side'] == 'LONG':
        gorl = (current_price - o['price']) * o['amount'] * leverage
    elif o['side'] == 'SHORT':
        gorl = (o['price'] - current_price) * o['amount'] * leverage
    else:
        gorl = 0
        print('UNKNOWN order["side"]', o)
        sys.exit()

    capital += (o['price'] * o['amount'] + gorl)
    commision = o['money_for_pair'] * 0.0006 * leverage
    capital -= commision

    o['gorl'] = round(gorl, 2)
    o['pnl'] = round(gorl / (o['price'] * o['amount']), 2)
    o['sell_date'] = index
    o['day'] = (o['sell_date'] - o['buy_date']).days
    o['sell_zscore'] = round(zscore_df[pair_name][index], 2)
    o['reason'] = reason
    number_of_transactions += 1
    if print_log:
        print(index, 'FLAT', pair_name, o)


def calculate_portfolio(index):
    money_on_stocks = 0
    global capital

    for pair_name, order_pair in orders.items():  # Elde kalan malları sat
        for o in order_pair:
            current_price = df[o['stock']][index]
            if o['side'] == 'LONG':
                gorl = (current_price - o['price']) * o['amount']
            elif o['side'] == 'SHORT':
                gorl = (o['price'] - current_price) * o['amount']

            money_on_stocks += (o['price'] * o['amount'] + gorl)

    # print(index, money_on_stocks + capital)
    return money_on_stocks + capital


def get_money_for_stock_pair(index, money_for_pair, stock1, stock2):
    if money_for_pair == 0:
        return 0, 0
    # s1_vol = volatility[stock1][index]
    # s2_vol = volatility[stock2][index]
    #
    # s1_money = money_for_pair * s1_vol / (s1_vol + s2_vol)
    # s2_money = money_for_pair * s2_vol / (s1_vol + s2_vol)
    # if s1_money is None or s2_money is None:
    s1_money = money_for_pair / 2
    s2_money = money_for_pair / 2

    # print(s1_money, s1_vol,  s2_money, s2_vol)
    return s1_money, s2_money


def calculate_statistics(capital_df):
    # if self.print_result:
    daily_rets = capital_df.resample('B').last()

    daily_rets = daily_rets.ffill().pct_change().dropna()
    perf_func = pf.timeseries.perf_stats

    perf_stats = perf_func(returns=daily_rets,
                               factor_returns=None,
                               positions=None,
                               transactions=None,
                               turnover_denom="AGB")
    print(perf_stats)

"""
Market simülasyonunu başlat
"""
capital = 100000
orders = {}
allowed_pair_count = 10
leverage = 6
max_day_for_position = 12
zscore_limit_for_flat = 1.5    # S2-S1 zscore olunce en iyi kombinasyon 0.50, 0.90
zscore_limit_for_position = 2
print_log = True
number_of_transactions = 0
stop_enabled = True
stop_dist = 0.50
capital_history = pd.Series(index=zscore_df.index, dtype=float)

for index in zscore_df.index:
    pairs_list = get_pairs_at_index(pairs_df, index)
    # print(index, pairs_list)
    capital_history[index] = calculate_portfolio(index)
    pair_len = len(pairs_list)
    order_len = len(orders)
    print(index, pair_len, order_len, int(capital_history[index]))
    if pair_len > 0:
        if order_len >= allowed_pair_count or pair_len == order_len:
            cnt = 0
        elif pair_len > order_len:
            cnt = pair_len - order_len
        elif order_len > pair_len:
            cnt = 0
            for p in orders.keys():  # Emir olmayan orderleri hesaba ekle
                if p in pairs_list:
                    continue
                else:
                    cnt += 1
        if cnt == 0:
            money_for_pair = 0
        else:
            money_for_pair = capital / cnt
    else:
        money_for_pair = 0

    # print(index, 'order_len:', order_len, 'pair_len:', pair_len, 'new_count:', cnt, 'money_for_pair:', int(money_for_pair), 'total_cap:', int(capital))
    # print(index, int(money_for_pair), cnt)
    remove_pairs = []
    for pair_name in orders.keys():
        if abs(zscore_df[pair_name][index]) < zscore_limit_for_flat:
            flat(pair_name, orders[pair_name][0], reason='score_limit')
            flat(pair_name, orders[pair_name][1], reason='score_limit')
            remove_pairs.append(pair_name)
        elif (index - orders[pair_name][0]['buy_date']).days >= max_day_for_position:
            # print((index - orders[pair_name][0]['buy_date']).days)
            flat(pair_name, orders[pair_name][0], reason='day_limit')
            flat(pair_name, orders[pair_name][1], reason='day_limit')
            remove_pairs.append(pair_name)

        elif stop_enabled:
            current_zscore = zscore_df[pair_name][index]
            if orders[pair_name][0]['buy_zscore'] < 0:
                if current_zscore < orders[pair_name][0]['stop_zscore']:
                    flat(pair_name, orders[pair_name][0], reason='stop_zscore')
                    flat(pair_name, orders[pair_name][1], reason='stop_zscore')
                    remove_pairs.append(pair_name)
            else:
                if current_zscore > orders[pair_name][0]['stop_zscore']:
                    flat(pair_name, orders[pair_name][0], reason='stop_zscore')
                    flat(pair_name, orders[pair_name][1], reason='stop_zscore')
                    remove_pairs.append(pair_name)

    for p in remove_pairs:
        del orders[p]

    for pair_name in pairs_list:
        stock1, stock2 = pair_name.split('-')
        # else:   # ilk kez al sat sinyali üretecek
        if zscore_df[pair_name][index] < -1 * zscore_limit_for_position:
            if pair_name in orders:
                continue
            current_zscore = round(zscore_df[pair_name][index], 2)
            c_stopscore = current_zscore - stop_dist
            s1_money, s2_money = get_money_for_stock_pair(index, money_for_pair, stock1, stock2)
            amount1 = int(s1_money / df[stock1][index])
            amount2 = int(s2_money / df[stock2][index])

            if amount1 == 0 or amount2 == 0:
                # print('There is no money for pair:', pair_name)
                continue
            o1 = {'stock': stock1, 'buy_date': index, 'side': 'SHORT', 'price': df[stock1][index],
                  'amount': amount1, 'money_for_pair': money_for_pair/2,
                  'buy_zscore': current_zscore, 'stop_zscore': c_stopscore}
            o2 = {'stock': stock2, 'buy_date': index, 'side': 'LONG', 'price': df[stock2][index],
                  'amount': amount2, 'money_for_pair': money_for_pair/2,
                  'buy_zscore': current_zscore, 'stop_zscore': c_stopscore}
            capital -= (o1['amount'] * o1['price'] + o2['amount'] * o2['price'])
            orders[stock1 + '-' + stock2] = [o1, o2]
            # print(index, pair_name, o1)
            # print(index, pair_name, o2)
            # print(index, pair_name, 'zscore[index] < -1')
            # LONG s1, short s2
            pass
        elif zscore_df[pair_name][index] > zscore_limit_for_position:
            if pair_name in orders:
                continue

            current_zscore = round(zscore_df[pair_name][index], 2)
            c_stopscore = current_zscore + stop_dist

            s1_money, s2_money = get_money_for_stock_pair(index, money_for_pair, stock1, stock2)
            amount1 = int(s1_money / df[stock1][index])
            amount2 = int(s2_money / df[stock2][index])
            if amount1 == 0 or amount2 == 0:
                # print('There is no money for pair:', pair_name)
                continue

            o1 = {'stock': stock1, 'buy_date': index, 'side': 'LONG', 'price': df[stock1][index],
                  'amount': amount1, 'money_for_pair': money_for_pair/2,
                  'buy_zscore': current_zscore, 'stop_zscore': c_stopscore}
            o2 = {'stock': stock2, 'buy_date': index, 'side': 'SHORT', 'price': df[stock2][index],
                  'amount': amount2, 'money_for_pair': money_for_pair/2,
                  'buy_zscore': current_zscore, 'stop_zscore': c_stopscore}
            capital -= (o1['amount'] * o1['price'] + o2['amount'] * o2['price'])
            orders[stock1 + '-' + stock2] = [o1, o2]
            # print(index, pair_name, o1)
            # print(index, pair_name, o2)
            # print(index, pair_name, 'zscore[index] > 1')
            # SHORT s1, LONG s2
            pass

for pair_name, order_pair in orders.items():    # Elde kalan malları sat
    for o in order_pair:
        flat(pair_name, o, reason='end_of_sim')

calculate_statistics(capital_history)
capital_history.plot()
print(round(capital, 2), number_of_transactions)
plt.show()

"""
2005-2015 arası genellikle iyi çalışıyor
2015'den sonra kötü çalışmaya başlıyor. statinoray test de ekledim pair selectiona, yine işe yaramadı

BIST_BANK denedim sadece pair selection için yine işe yaramadı.
stationry check ve pair_selection tresholdlarıyla oynadım yine işe yaramadı

Sıkıldım bu stratejiden
Saatlikte de düzgün çalışmadı sanırım.
Bir ara tekrar bakarım ama keyif almıyorum bundan şimdilik
Bir de money kısmında sorun var marketin, capitalde ani düşüşler oluyor. Onu düzeltmek lazım, bakasım yok
(KOZAA'Daki korkunç artış yapmış onu, stop koymak lazım kesin)
2020-11-04
"""
