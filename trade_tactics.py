#!/usr/bin/env python
# coding: utf-8
import streamlit as st

import datetime
import pprint
import requests
import yaml
# from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.markdown(f'<p><span style = "font-size:24px"><strong>交易策略分析</strong></span></p>',unsafe_allow_html=True)
st.markdown('查詢單一股票買賣點')
查詢股票 = st.text_input("輸入查詢股票(可查詢台/美股如0050、AAPL)")
try:
    if 查詢股票[0].isdigit():
        查詢股票 += '.TW'
except:
    pass
查詢期間 = st.number_input("輸入查詢期間(月)(如12代表1年)",value=12)

PERIOD_TYPE_DAY = 'day'
PERIOD_TYPE_WEEK = 'week'
PERIOD_TYPE_MONTH = 'month'
PERIOD_TYPE_YEAR = 'year'

# Valid frequencies: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
FREQUENCY_TYPE_MINUTE = 'm'
FREQUENCY_TYPE_HOUR = 'h'
FREQUENCY_TYPE_DAY = 'd'
FREQUENCY_TYPE_WEEK = 'wk'
FREQUENCY_TYPE_MONTH = 'mo'




class Share(object):

    def __init__(self, symbol):
        self.symbol = symbol


    def get_historical(self, period_type, period, frequency_type, frequency):
        data = self._download_symbol_data(period_type, period,
                                          frequency_type, frequency)

        valid_frequency_types = [
            FREQUENCY_TYPE_MINUTE, FREQUENCY_TYPE_HOUR, FREQUENCY_TYPE_DAY,
            FREQUENCY_TYPE_WEEK, FREQUENCY_TYPE_MONTH
        ]

        if frequency_type not in valid_frequency_types:
            raise ValueError('Invalid frequency type: ' % frequency_type)

        # for i in range(len(data['timestamp'])):
        #     if i < (len(data['timestamp']) - 1):
        #         print(datetime.datetime.utcfromtimestamp(
        #                 data['timestamp'][i + 1]
        #             ).strftime('%Y-%m-%d %H:%M:%S'),
        #             data['timestamp'][i + 1] - data['timestamp'][i]
        #         )

        if 'timestamp' not in data:
            return None

        return_data = {
            'timestamp': [x * 1000 for x in data['timestamp']],
            'open': data['indicators']['quote'][0]['open'],
            'high': data['indicators']['quote'][0]['high'],
            'low': data['indicators']['quote'][0]['low'],
            'close': data['indicators']['quote'][0]['close'],
            'adjclose': data['indicators']['adjclose'][0]['adjclose'],
            'volume': data['indicators']['quote'][0]['volume']
        }
        #print(return_data)

        return return_data


    def _set_time_frame(self, period_type, period):
        now = datetime.datetime.now()

        if period_type == PERIOD_TYPE_DAY:
            period = min(period, 59)
            start_time = now - datetime.timedelta(days=period)
        elif period_type == PERIOD_TYPE_WEEK:
            period = min(period, 59)
            start_time = now - datetime.timedelta(days=period * 7)
        elif period_type == PERIOD_TYPE_MONTH:
            period = min(period, 59)
            start_time = now - datetime.timedelta(days=period * 30)
        elif period_type == PERIOD_TYPE_YEAR:
            period = min(period, 59)
            start_time = now - datetime.timedelta(days=period * 365)
        else:
            raise ValueError('Invalid period type: ' % period_type)

        end_time = now

        return int(start_time.timestamp()), int(end_time.timestamp())


    def _download_symbol_data(self, period_type, period,
                              frequency_type, frequency):
        start_time, end_time = self._set_time_frame(period_type, period)
        url = (
            'https://query1.finance.yahoo.com/v8/finance/chart/{0}?symbol={0}'
            '&period1={1}&period2={2}&interval={3}&'
            'includePrePost=true&events=div%7Csplit%7Cearn&lang=en-US&'
            'region=US&crumb=t5QZMhgytYZ&corsDomain=finance.yahoo.com'
        ).format(self.symbol, start_time, end_time,
                 self._frequency_str(frequency_type, frequency))

        resp_json = requests.get(url).json()

        if self._is_yf_response_error(resp_json):
            self._raise_yf_response_error(resp_json)
            return

        data_json = resp_json['chart']['result'][0]

        return data_json


    def _is_yf_response_error(self, resp):
        return resp['chart']['error'] is not None


    def _raise_yf_response_error(self, resp):
        # raise YahooFinanceError(
        raise Error(
            '{0}: {1}'.format(
                resp['chart']['error']['code'],
                resp['chart']['error']['description']
            )
        )

    def _frequency_str(self, frequency_type, frequency):
        return '{1}{0}'.format(frequency_type, frequency)


def get_historical_quote(symbol, months):
    share_object = Share(symbol)
    # get historical data 
    symbol_data = share_object.get_historical(PERIOD_TYPE_MONTH,months,FREQUENCY_TYPE_DAY, 1)
    df=pd.DataFrame(symbol_data)
    # create date from timestamp column 
    df['date'] = pd.to_datetime(df['timestamp'],unit='ms').apply(lambda x: x.date())
    # drop timestamp column 
    df.drop(['timestamp'], axis=1, inplace=True)
    # move date column to first column
    new_sequence = [df.columns[-1]] + [x for x in df.columns[:-1]]
    df = df[new_sequence] 
    if symbol.endswith('.TW'):
        df[['open', 'close', 'high', 'low', 'adjclose']] = df[['open', 'close', 'high', 'low', 'adjclose']].round(2)
        # df['volume'] = df['volume'].astype(np.int64)
        # df['open']= np.around(pd['open'], 2)
        # df['close']= np.around(pd['close'], 2)
        # df['high']= np.around(pd['high'], 2)
        # df['low']= np.around(pd['low'], 2)
        # df['adjclose']= np.around(pd['adjclose'], 2)
    return df

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


start = st.button("<<START>>")
if start:
    df = get_historical_quote(查詢股票 , 查詢期間)
    st.dataframe(df.tail(5))


    sma_short = pd.DataFrame()
    sma_short['date'] = df['date']
    sma_short['adjclose'] = df['adjclose'].rolling(window=22).mean()
    #sma_short.loc[15:30]


    sma_long = pd.DataFrame()
    sma_long['date'] = df['date']
    sma_long['adjclose'] = df['adjclose'].rolling(window=66).mean()
    #sma_long.loc[60:180]

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
    sns.lineplot(x='date', y='adjclose', data=sma_short, color='g', label='short_term')
    sns.lineplot(x='date', y='adjclose', data=sma_long, color='b', label='long_term')

    plt.plot(df['date'], df['adjclose'], color='r', alpha=0.5, label='date_price')
    plt.scatter(df['date'], signal_buy, c='r', marker='^', s=150)
    plt.scatter(df['date'], signal_sell, c='g', marker='^', s=150)

    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    # 計算損益(profit/loss)
    # handling_fee：手續費率
    # allow_oversold：是否允許超賣

    def calc_profit(df_buy, df_sell, df, handling_fee=0.0, allow_oversold = False):
        買入=[]
        賣出=[]
        交易日=[]
        庫存=[]
        損益=[]
        df_profit = df_buy.merge(df_sell, on='date', how='outer') 
        df_profit.sort_values(by='date', inplace=True)
        df_date = df.set_index('date')
        balance=0
        profit=0
        cost=0
        buy_ = 0
        sell_ = 0
        for index, row in df_profit.iterrows():
            if not np.isnan(row['signal_buy']):
                balance+=1
                cost+=df_date.loc[row['date'], 'adjclose']
                交易日.append(str(row['date'].year)+'-'+str(row['date'].month)+'-'+str(row['date'].day))
                買入.append(df_date.loc[row['date'], 'adjclose'])
                賣出.append(0)
                庫存.append(balance)
                損益.append(0)
            elif not np.isnan(row['signal_sell']):
                if balance>0:
                    avg_cost = cost / balance
                    profit += df_date.loc[row['date'], 'adjclose']*(1-handling_fee) - avg_cost
                    balance-=1
                    交易日.append(str(row['date'].year)+'-'+str(row['date'].month)+'-'+str(row['date'].day))
                    買入.append(0)
                    賣出.append(df_date.loc[row['date'], 'adjclose']*(1-handling_fee))  
                    庫存.append(balance)
                    損益.append(df_date.loc[row['date'], 'adjclose']*(1-handling_fee) - avg_cost)
                    cost -= avg_cost
                else: # 超賣
                    # 不允許超賣
                    if not allow_oversold:
                        continue    
                    profit += df_date.loc[row['date'], 'adjclose']*(1-handling_fee)
                    balance-=1


        if balance>0:   # 賣出平倉
            profit +=  df.iloc[-1]['adjclose']*(1-handling_fee) * balance - cost
            sell_ += ( df.iloc[-1]['adjclose']*(1-handling_fee))
            交易日.append(str(df.iloc[-1]['date'].year)+'-'+str(df.iloc[-1]['date'].month)+'-'+str(df.iloc[-1]['date'].day))
            買入.append(0)
            賣出.append( df.iloc[-1]['adjclose']*(1-handling_fee))
            損益.append( df.iloc[-1]['adjclose']*(1-handling_fee) * balance - cost)
            balance -=1
            庫存.append(balance)

        elif balance<0: # 買進平倉
            profit += df_date.loc[row['date'], 'adjclose'] 
        trade_df = pd.DataFrame({'交易日':交易日,'買入價':買入,'賣出價':賣出,'庫存':庫存,'損益':損益})
        return profit, trade_df


    ttprofit, trade_df = calc_profit(df_buy, df_sell, df, 0.003, False)
    st.dataframe(trade_df)
    st.markdown('期間總獲利(每次交易1單位) %.2f'%(ttprofit))


