#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import my_share 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = my_share.get_historical_quote('TSLA', 24)



sma_short = pd.DataFrame()
sma_short['date'] = df['date']
sma_short['adjclose'] = df['adjclose'].rolling(window=22).mean()
sma_short.loc[15:30]


sma_long = pd.DataFrame()
sma_long['date'] = df['date']
sma_long['adjclose'] = df['adjclose'].rolling(window=66).mean()
sma_long.loc[60:180]





plt.figure(figsize=(10,6))
sns.lineplot(x='date', y='adjclose', data=sma_short, color='gold')
sns.lineplot(x='date', y='adjclose', data=sma_long)



def buy_sell(df):
    signal_buy = []  # 買點價格
    signal_sell = [] # 賣點價格
    
    flag=-1          # 買賣點旗標，短期超過長期為1，反之為0
    
    # 掃描每一筆資料
    for index, row in df.iterrows():
        # 短期超過長期
        if row[df.columns[1]] > row[df.columns[2]]:
            if flag!=1: # 之前的短期未超過長期，即黃金交叉
                signal_buy.append(row[df.columns[3]])
                signal_sell.append(np.nan)
                flag=1
            else:
                signal_buy.append(np.nan)
                signal_sell.append(np.nan)
        elif row[df.columns[1]] < row[df.columns[2]]:
            if flag!=0: # 之前的長期未超過短期，即死亡交叉
                signal_buy.append(np.nan)
                signal_sell.append(row[df.columns[3]])
                flag=0
            else:
                signal_buy.append(np.nan)
                signal_sell.append(np.nan)
        else:
            signal_buy.append(np.nan)
            signal_sell.append(np.nan)
    return (signal_buy, signal_sell)


# 合併短期與長期移動平均線
df_new = sma_short.copy()
df_new = df_new.rename({'adjclose':'sma_short'}, axis=1)
df_new.insert(2, 'sma_long', sma_long['adjclose'])
df_new.insert(3, 'adjclose', df['adjclose'])


signal_buy, signal_sell = buy_sell(df_new)
# 買點
df_buy = pd.DataFrame({'date': df['date'], 'signal_buy':signal_buy})
df_buy = df_buy[~np.isnan(signal_buy)]


# 賣點
df_sell = pd.DataFrame({'date': df['date'], 'signal_sell':signal_sell})
df_sell = df_sell[~np.isnan(signal_sell)]



plt.figure(figsize=(10,6))
sns.lineplot(x='date', y='adjclose', data=sma_short, color='g', label='短期趨勢')
sns.lineplot(x='date', y='adjclose', data=sma_long, color='b', label='長期趨勢')

plt.plot(df['date'], df['adjclose'], color='r', alpha=0.5, label='日線')
plt.scatter(df['date'], signal_buy, c='r', marker='^', s=150)
plt.scatter(df['date'], signal_sell, c='g', marker='^', s=150)

plt.legend()



# 計算損益(profit/loss)
def calc_profit(df_buy, df_sell, df):
    df_profit = df_buy.merge(df_sell, on='date', how='outer') 
    df_profit.sort_values(by='date', inplace=True)

    df_date = df.set_index('date')

    balance=0
    profit=0
    cost=0
    for index, row in df_profit.iterrows():
        if not row['signal_buy'] is None:
            balance+=1
            cost+=df_date.loc[row['date'], 'adjclose']
        elif not row['signal_sell'] is None:
            if balance>0:
                avg_cost = cost / balance
                profit += df_date.loc[row['date'], 'adjclose'] - avg_cost
                cost -= avg_cost
            else:
                profit += df_date.loc[row['date'], 'adjclose']

            balance-=1

    if balance>0:
        profit += df_date.loc[row['date'], 'adjclose'] * balance - cost
    elif balance<0:
        profit += df_date.loc[row['date'], 'adjclose'] * balance
    
    return profit
    
calc_profit(df_buy, df_sell, df)    



