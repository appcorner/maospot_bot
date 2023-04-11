# -*- coding: utf-8 -*-

from asyncio import get_event_loop, gather, sleep
import numpy as np
import pandas as pd
import pandas_ta as ta
import time
import mplfinance as mpf 
import matplotlib.pyplot as plt
from LineNotify import LineNotify
import config
import os
import pathlib
import logging
from logging.handlers import RotatingFileHandler
from random import randint, shuffle
from datetime import datetime
import json
from uuid import uuid4

# -----------------------------------------------------------------------------
# API_KEY, API_SECRET, LINE_NOTIFY_TOKEN in config.ini
# -----------------------------------------------------------------------------

# import ccxt.async_support as ccxt
import ccxt_bk.async_support.bitkub as ccxt_bk

# print('CCXT Version:', ccxt.__version__)
# -----------------------------------------------------------------------------

bot_name = 'MaoMao'
bot_vesion = '1.0'

bot_fullname = f'{bot_name} Spot (Bitkub) version {bot_vesion}'

# ansi escape code
CLS_SCREEN = '\033[2J\033[1;1H' # cls + set top left
CLS_LINE = '\033[0J'
SHOW_CURSOR = '\033[?25h'
HIDE_CURSOR = '\033[?25l'
CRED  = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW  = '\33[33m'
CEND = '\033[0m'
CBOLD = '\33[1m'

# กำหนดเวลาที่ต้องการเลื่อนการอ่านข้อมูล เป็นจำนวนวินาที
TIME_SHIFT = config.TIME_SHIFT

TIMEFRAME_SECONDS = {
    '1m': 60,
    '5m': 60*5,
    '15m': 60*15,
    '30m': 60*30,
    '1h': 60*60,
    '4h': 60*60*4,
    '1d': 60*60*24,
}

CANDLE_PLOT = config.CANDLE_PLOT
CANDLE_SAVE = 1 + max(config.mid_value,config.slow_value,config.MACD_SLOW,config.RSI_PERIOD,config.rolling_period)
CANDLE_LIMIT = max(config.CANDLE_LIMIT,CANDLE_SAVE)

UB_TIMER_SECONDS = [
    TIMEFRAME_SECONDS[config.timeframe],
    15,
    20,
    30,
    60,
    int(TIMEFRAME_SECONDS[config.timeframe]/2)
]

BALANCE_COLUMNS = ["asset", "free", "locked", "total"]
# BALANCE_COLUMNS_RENAME = ["Asset", "Free", "Locked", "Total", "Symbol", "Market Price", "Unrealized Price"]
# BALANCE_COLUMNS_DISPLAY = ["Symbol", "Free", "Locked", "Total", "Symbol", "Market Price", "Unrealized Price"]
BALANCE_COLUMNS_DISPLAY = ["asset", "free", "locked", "total", "marketPrice", "Margin", "unrealizedProfit"]

CSV_COLUMNS = [
        "symbol", "signal_index", "margin_type",
        "trade_mode", "trade_long", "trade_short",
        "leverage", "cost_type", "cost_amount",
        "tpsl_mode",
        "tp_long", "tp_short",
        "tp_close_long", "tp_close_short",
        "sl_long", "sl_short",
        "trailing_stop_mode",
        "callback_long", "callback_short",
        "active_tl_long", "active_tl_short",
        "fast_type",
        "fast_value",
        "mid_type",
        "mid_value",
        "slow_type",
        "slow_value"
        ]

DATE_SUFFIX = datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------------------------------------------------------
# global variable
# ----------------------------------------------------------------------------
notify = LineNotify(config.LINE_NOTIFY_TOKEN)

all_positions = pd.DataFrame(columns=BALANCE_COLUMNS)
count_trade = 0

start_balance_total = 0.0
balance_total = 0.0
balance_entry = {}

watch_list = []
all_symbols = {}
all_candles = {}

orders_history = {}

total_risk = {}
is_send_notify_risk = False

is_positionside_dual = False

is_send_notify_error = True
last_error_message = ''

symbols_setting = pd.DataFrame(columns=CSV_COLUMNS)

history_file_csv = 'orders_history.csv'
history_json_path = 'orders_history.json'

async def getExchange():
    # exchange = ccxt.binance({
    #     "apiKey": config.API_KEY,
    #     "secret": config.API_SECRET,
    #     "options": {"defaultType": "future"},
    #     "enableRateLimit": True
    #     })
    # if config.SANDBOX:
    #     exchange.set_sandbox_mode(True)
    exchange = ccxt_bk.bitkub({
        "apiKey": config.API_KEY,
        "secret": config.API_SECRET,
        "options": {"defaultType": "spot"},
        "enableRateLimit": True
        })
    
    return exchange

async def retry(func, limit=0, wait_s=3, wait_increase_ratio=2):
    attempt = 1
    while True:
        try:
            return await func()
        except Exception as ex:
            if 'RequestTimeout' not in type(ex).__name__:
                raise ex
            if 0 < limit <= attempt:
                logger.warning("no more attempts")
                raise ex

            logger.error("failed execution attempt #%d", attempt)

            attempt += 1
            logger.info("waiting %d s before attempt #%d", wait_s, attempt)
            time.sleep(wait_s)
            wait_s *= wait_increase_ratio

def school_round(a_in,n_in):
    ''' python uses "banking round; while this round 0.05 up '''
    if (a_in * 10 ** (n_in + 1)) % 10 == 5:
        return round(a_in + 1 / 10 ** (n_in + 1), n_in)
    else:
        return round(a_in, n_in)

def amount_to_precision(symbol, amount_value):
    amount_precision = all_symbols[symbol]['amount_precision']
    amount = school_round(amount_value, amount_precision)
    return amount
def price_to_precision(symbol, price_value):
    price_precision = all_symbols[symbol]['price_precision']
    price = school_round(price_value, price_precision)
    return price

def detect_sideway_trend(df, atr_multiple=1.5, n=15, mode='2'):
    sw_df = df.copy()

    # Calculate the Bollinger Bands
    sw_df.ta.bbands(close='Close', length=n, append=True)
    bb_sfx = f'_{n}_2.0'
    # columns = {f"BBL{bb_sfx}": "BBL", f"BBM{bb_sfx}": "BBM", f"BBU{bb_sfx}": "BBU", f"BBB{bb_sfx}": "BBB", f"BBP{bb_sfx}": "BBP"}
    # sw_df.rename(columns=columns, inplace = True)
    
    # Check if the current price is within the Bollinger Bands
    # inBB = sw_df[['close', 'BBL', 'BBU']].apply(lambda x: (1 if x['close'] > x['BBL'] and x['close'] < x['BBU'] else 0), axis=1)
    inBB = sw_df[['close', f'BBL{bb_sfx}', f'BBU{bb_sfx}']].apply(lambda x: (1 if x['close'] > x[f'BBL{bb_sfx}'] and x['close'] < x[f'BBU{bb_sfx}'] else 0), axis=1)
    sw_df['inBB'] = inBB
    
    # Calculate the MACD
    # sw_df.ta.macd(close='close', append=True)
    # macd_sfx = '_12_26_9'
    # # columns = {f"MACD{macd_sfx}": "MACD", f"MACDs{macd_sfx}": "MACDs", f"MACDh{macd_sfx}": "MACDh"}
    # # sw_df.rename(columns=columns, inplace = True)
    
    # Check if the MACD histogram is positive
    MACDp = sw_df[['MACDh']].apply(lambda x: (1 if x['MACDh'] > 0 else 0), axis=1)
    # MACDp = sw_df[[f'MACDh{macd_sfx}']].apply(lambda x: (1 if x[f'MACDh{macd_sfx}'] > 0 else 0), axis=1)
    sw_df['MACDp'] = MACDp

    # Calculate the rolling average of the high and low prices
    avehigh = sw_df['high'].rolling(n).mean()
    avelow = sw_df['low'].rolling(n).mean()
    avemidprice = (avehigh + avelow) / 2

    # get upper and lower bounds to compare to period highs and lows
    high_low = sw_df['high'] - sw_df['low']
    high_close = np.abs(sw_df['high'] - sw_df['close'].shift())
    low_close = np.abs(sw_df['low'] - sw_df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr14 = true_range.rolling(14).sum()/14

    sw_df['UPB'] = avemidprice + atr_multiple * atr14
    sw_df['LPB'] = avemidprice - atr_multiple * atr14

    # get the period highs and lows
    rangemaxprice = sw_df['high'].rolling(n).max()
    rangeminprice = sw_df['low'].rolling(n).min()

    # Calculate the sideways range using vectorized operations
    sideways = np.where((rangemaxprice < sw_df['UPB']) & (rangemaxprice > sw_df['LPB']) & (rangeminprice < sw_df['UPB']) & (rangeminprice > sw_df['LPB']), 1, 0)
    sw_df['sideways'] = sideways

    # Return 1 if the current price is within the Bollinger Bands, the MACD histogram is positive, and the trend is sideways, otherwise return 0
    def sideways_range(in_bb, macd_p, sideways):
      if in_bb and macd_p and sideways == 1:
          return 1
      else:
          return 0

    sideways_bb_macd = sw_df[['inBB', 'MACDp', 'sideways']].apply(lambda x: sideways_range(x['inBB'], x['MACDp'], x['sideways']), axis=1)

    del sw_df

    if mode == '1':
        return sideways
    else:
        return sideways_bb_macd

def cal_callback_rate(symbol, closePrice, targetPrice):
    rate = round(abs(closePrice - targetPrice) / closePrice * 100.0, 1)
    logger.debug(f'{symbol} closePrice:{closePrice}, targetPrice:{targetPrice}, callback_rate:{rate}')
    if rate < 0.1:
        return 0.1
    else:
        return rate

def cal_minmax_fibo(symbol, df, closePrice=0.0):
    iday = df.tail(CANDLE_PLOT)

    # swing low
    periods = 3
    lows_list = list(iday['low'])
    lows_list.reverse()
    # logger.debug(lows_list)
    # swing_low = lows_list[0]
    swing_lows = []
    for i in range(len(lows_list)):
        if i >= periods:
            # Check if the current price is the lowest in the last `periods` periods
            if min(lows_list[i-periods:i+1]) == lows_list[i]:
                swing_lows.append(lows_list[i])
    # logger.debug(swing_lows)

    signalIdx = config.signal_index
    if symbol in symbols_setting.index:
        signalIdx = int(symbols_setting.loc[symbol]['signal_index'])

    iday_minmax = iday[:CANDLE_PLOT+signalIdx]
    minimum_index = iday_minmax['low'].idxmin()
    minimum_price = iday_minmax['low'].min()
    maximum_index = iday_minmax['high'].idxmax()
    maximum_price = iday_minmax['high'].max()
    #Calculate the max high and min low price
    difference = maximum_price - minimum_price #Get the difference

    # fibo_values = [0,0.1618,0.236,0.382,0.5,0.618,0.786,1,1.382]
    fibo_values = [0,0.236,0.382,0.5,0.618,0.786,1,1.382]

    isFiboRetrace = True
    minmax_points = []
    fibo_levels = []
    periods = config.SWING_TF
    swing_lows = []
    swing_highs = []
    tp = 0.0
    sl = 0.0

    # logger.debug(minimum_index)
    # logger.debug(maximum_index)

    # iday_minmax['sw_low'] = np.nan
    # iday_minmax['sw_high'] = np.nan
    for i in range(len(iday_minmax)):
        if i >= periods:
            if min(iday_minmax['low'].iloc[i-periods:i+1+periods]) == iday_minmax['low'].iloc[i]:
                swing_lows.append(iday_minmax['low'].iloc[i])
                # iday_minmax['sw_low'].iloc[i] =  iday_minmax['low'].iloc[i]
            if max(iday_minmax['high'].iloc[i-periods:i+1+periods]) == iday_minmax['high'].iloc[i]:
                swing_highs.append(iday_minmax['high'].iloc[i])
                # iday_minmax['sw_high'].iloc[i] =  iday_minmax['low'].iloc[i]

    isFiboRetrace = datetime.strptime(str(minimum_index), '%Y-%m-%d %H:%M:%S%z') > datetime.strptime(str(maximum_index), '%Y-%m-%d %H:%M:%S%z')
    # print(isFiboRetrace)

    if isFiboRetrace:
        minmax_points.append((maximum_index,maximum_price))
        minmax_points.append((minimum_index,minimum_price))
        for idx, fibo_val in enumerate(fibo_values):
            fibo_level = price_to_precision(symbol, minimum_price + difference * fibo_val)
            fibo_levels.append(fibo_level)
            if tp == 0.0 and closePrice < fibo_level:
                tp_fibo = min(idx+config.TP_FIBO, len(fibo_values)-1)
                tp = price_to_precision(symbol, minimum_price + difference * fibo_values[tp_fibo])
    else:
        # maxidx = np.where(iday_minmax.index==maximum_index)[0][0]
        maxidx = iday_minmax.index.get_loc(maximum_index)
        # print(maxidx)
        if maxidx < len(iday_minmax)-1:
            new_minimum_index = iday_minmax['low'].iloc[maxidx+1:].idxmin()
            new_minimum_price = iday_minmax['low'].iloc[maxidx+1:].min()
        else:
            new_minimum_index = iday_minmax['low'].iloc[maxidx:].idxmin()
            new_minimum_price = iday_minmax['low'].iloc[maxidx:].min()
        minmax_points.append((minimum_index,minimum_price))
        minmax_points.append((maximum_index,maximum_price))
        minmax_points.append((new_minimum_index,new_minimum_price))
        for idx, fibo_val in enumerate(fibo_values):
            fibo_level = price_to_precision(symbol, new_minimum_price + difference * fibo_val)
            fibo_levels.append(fibo_level)
            if tp == 0.0 and closePrice < fibo_level:
                tp_fibo = min(idx+config.TP_FIBO, len(fibo_values)-1)
                tp = price_to_precision(symbol, new_minimum_price + difference * fibo_values[tp_fibo])

    sl_fibo = closePrice - difference * fibo_values[1]
    sl_sw = min(swing_lows[-config.SWING_TEST:])
    sl = min(sl_fibo, sl_sw)

    # fixed tp by sl ratio 1:2
    if tp == 0.0:
        tp = price_to_precision(symbol, closePrice + (closePrice - sl) * 2.0)

    if config.CB_AUTO_MODE == 1:
        callback_rate = cal_callback_rate(symbol, closePrice, tp)
    else:
        callback_rate = cal_callback_rate(symbol, closePrice, sl)

    return {
        'fibo_type': 'retractment' if isFiboRetrace else 'extension',
        'difference': difference,
        'min_max': minmax_points, 
        'fibo_values': fibo_values,
        'fibo_levels': fibo_levels,
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'tp': tp,
        'sl': sl,
        'tp_txt': '-',
        'sl_txt': '-',
        'callback_rate': callback_rate
    }

async def line_chart(symbol, df, msg, pd='', fibo_data=None, **kwargs):
    if config.strategy_mode == 'ADXRSI':
        line_chart_adxrsi(symbol, df, msg, pd, fibo_data, **kwargs)
    else:
        line_chart_ema(symbol, df, msg, pd, fibo_data, **kwargs)
def line_chart_ema(symbol, df, msg, pd='', fibo_data=None, **kwargs):
    try:
        print(f"{symbol} create line_chart")
        data = df.tail(CANDLE_PLOT)

        data_len = data.shape[0]
        RSI30 = [30 for i in range(0, data_len)]
        RSI50 = [50 for i in range(0, data_len)]
        RSI70 = [70 for i in range(0, data_len)]

        showFibo = fibo_data != None and 'exit' not in pd.lower()

        colors = ['green' if value >= 0 else 'red' for value in data['MACDh']]
        added_plots = [
            mpf.make_addplot(data['fast'],color='red',width=0.5),
            mpf.make_addplot(data['mid'],color='orange',width=0.5),
            mpf.make_addplot(data['slow'],color='green',width=0.5),

            mpf.make_addplot(data['RSI'],ylim=(10, 90),panel=2,color='blue',width=0.75,
                ylabel=f"RSI ({config.RSI_PERIOD})", y_on_right=False),
            mpf.make_addplot(RSI30,ylim=(10, 90),panel=2,color='red',linestyle='-.',width=0.5),
            mpf.make_addplot(RSI50,ylim=(10, 90),panel=2,color='red',linestyle='-.',width=0.5),
            mpf.make_addplot(RSI70,ylim=(10, 90),panel=2,color='red',linestyle='-.',width=0.5),

            mpf.make_addplot(data['MACDh'],type='bar',width=0.5,panel=3,color=colors,
                ylabel=f"MACD ({config.MACD_FAST})", y_on_right=True),
            mpf.make_addplot(data['MACD'],panel=3,color='orange',width=0.75),
            mpf.make_addplot(data['MACDs'],panel=3,color='blue',width=0.75),
        ]

        kwargs = dict(
            figscale=1.2,
            figratio=(8, 7),
            panel_ratios=(8,2,2,2),
            addplot=added_plots,
            # tight_layout=True,
            # scale_padding={'left': 0.5, 'top': 2.5, 'right': 2.5, 'bottom': 0.75},
            scale_padding={'left': 0.5, 'top': 0.6, 'right': 1.0, 'bottom': 0.5},
            )

        fibo_title = ''

        if showFibo:
            fibo_colors = ['red','brown','orange','gold','green','blue','gray','purple','purple','purple']
            logger.debug(fibo_data)
            # fibo_colors.append('g')
            # fibo_data['fibo_levels'].append(fibo_data['swing_highs'][0])
            # fibo_colors.append('r')
            # fibo_data['fibo_levels'].append(fibo_data['swing_lows'][0])
            fibo_lines = dict(
                hlines=fibo_data['fibo_levels'],
                colors=fibo_colors,
                alpha=0.5,
                linestyle='-.',
                linewidths=1,
                )
            tpsl_colors = ['g','r']
            tpsl_data = [fibo_data['tp'], fibo_data['sl']]
            tpsl_lines = dict(
                hlines=tpsl_data,
                colors=tpsl_colors,
                alpha=0.5,
                linestyle='-.',
                linewidths=1,
                )
            minmax_lines = dict(
                alines=fibo_data['min_max'],
                colors='black',
                linestyle='--',
                linewidths=0.1,
                )
            fibo_title = ' fibo-'+fibo_data['fibo_type'][0:2]
            kwargs['hlines']=tpsl_lines
            kwargs['alines']=minmax_lines

        myrcparams = {'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':8}
        mystyle = mpf.make_mpf_style(base_mpf_style='charles',rc=myrcparams)

        filename = f"./plots/order_{symbol}.png"
        fig, axlist = mpf.plot(
            data,
            volume=True,volume_panel=1,
            **kwargs,
            type="candle",
            xrotation=0,
            ylabel='Price',
            style=mystyle,
            returnfig=True,
        )
        # print(axlist)
        ax1,*_ = axlist

        title = ax1.set_title(f'{symbol} {pd} ({config.timeframe} @ {data.index[-1]}{fibo_title})')
        title.set_fontsize(14)

        if showFibo:
            difference = fibo_data['difference']
            fibo_levels = fibo_data['fibo_levels']
            for idx, fibo_val in enumerate(fibo_data['fibo_values']):
                ax1.text(0,fibo_levels[idx] + difference * 0.02,f'{fibo_val}({fibo_levels[idx]})',fontsize=8,color=fibo_colors[idx],horizontalalignment='left')

            fibo_tp = fibo_data['tp']
            fibo_tp_txt = fibo_data['tp_txt']
            ax1.text(CANDLE_PLOT,fibo_tp - difference * 0.04,fibo_tp_txt,fontsize=8,color='g',horizontalalignment='right')
            fibo_sl = fibo_data['sl']
            fibo_sl_txt = fibo_data['sl_txt']
            ax1.text(CANDLE_PLOT,fibo_sl - difference * 0.04,fibo_sl_txt,fontsize=8,color='r',horizontalalignment='right')

        fig.savefig(filename)

        plt.close(fig)

        notify.Send_Image(msg, image_path=filename)
        # await sleep(2)
        if config.remove_plot:
            os.remove(filename)

    except Exception as ex:
        print(type(ex).__name__, symbol, str(ex))
        logger.exception(f'line_chart {symbol}')

    return
def line_chart_adxrsi(symbol, df, msg, pd='', fibo_data=None, **kwargs):
    try:
        print(f"{symbol} create line_chart")
        data = df.tail(CANDLE_PLOT)

        showFibo = fibo_data != None and 'exit' not in pd.lower()

        data_len = data.shape[0]
        ADXLine = [kwargs['ADXIn'] for i in range(0, data_len)]
        RSIlo = [kwargs['RSIlo'] for i in range(0, data_len)]
        RSIhi = [kwargs['RSIhi'] for i in range(0, data_len)]
        STOlo = [kwargs['STOlo'] for i in range(0, data_len)]
        STOhi = [kwargs['STOhi'] for i in range(0, data_len)]

        # colors = ['green' if value >= 0 else 'red' for value in data['MACD']]
        added_plots = [
            mpf.make_addplot(data['RSI'],ylim=(10, 90),panel=2,color='blue',width=0.75,
                fill_between=dict(y1=kwargs['RSIlo'], y2=kwargs['RSIhi'], color="orange"),
                ylabel=f"RSI ({config.RSI_PERIOD})", y_on_right=False),
            mpf.make_addplot(RSIlo,ylim=(10, 90),panel=2,color='red',linestyle='-.',width=0.5),
            mpf.make_addplot(RSIhi,ylim=(10, 90),panel=2,color='red',linestyle='-.',width=0.5),

            mpf.make_addplot(data['ADX'],ylim=(0, 90),panel=3,color='red',width=0.75,
                ylabel=f"ADX ({config.ADX_PERIOD})", y_on_right=True),
            mpf.make_addplot(ADXLine,ylim=(0, 90),panel=3,color='red',linestyle='-.',width=0.5),

            mpf.make_addplot(data['STOCHk'],ylim=(0, 100),panel=4,color='blue',width=0.75,
                fill_between=dict(y1=kwargs['STOlo'], y2=kwargs['STOhi'], color="orange"),
                ylabel=f"STO ({config.STO_K_PERIOD})", y_on_right=False),
            mpf.make_addplot(data['STOCHd'],ylim=(0, 100),panel=4,color='red',width=0.75),
            mpf.make_addplot(STOlo,ylim=(0, 100),panel=4,color='red',linestyle='-.',width=0.5),
            mpf.make_addplot(STOhi,ylim=(0, 100),panel=4,color='red',linestyle='-.',width=0.5),
        ]

        kwargs = dict(
            figscale=1.2,
            figratio=(8, 7),
            panel_ratios=(8,1,2,1,2),
            addplot=added_plots,
            # tight_layout=True,
            # scale_padding={'left': 0.5, 'top': 2.5, 'right': 2.5, 'bottom': 0.75},
            scale_padding={'left': 0.5, 'top': 0.6, 'right': 1.0, 'bottom': 0.5},
            )

        fibo_title = ''

        if showFibo:
            fibo_colors = ['red','brown','orange','gold','green','blue','gray','purple','purple','purple']
            logger.debug(fibo_data)
            # fibo_colors.append('g')
            # fibo_data['fibo_levels'].append(fibo_data['swing_highs'][0])
            # fibo_colors.append('r')
            # fibo_data['fibo_levels'].append(fibo_data['swing_lows'][0])
            fibo_lines = dict(
                hlines=fibo_data['fibo_levels'],
                colors=fibo_colors,
                alpha=0.5,
                linestyle='-.',
                linewidths=1,
                )
            tpsl_colors = ['g','r']
            tpsl_data = [fibo_data['tp'], fibo_data['sl']]
            tpsl_lines = dict(
                hlines=tpsl_data,
                colors=tpsl_colors,
                alpha=0.5,
                linestyle='-.',
                linewidths=1,
                )
            minmax_lines = dict(
                alines=fibo_data['min_max'],
                colors='black',
                linestyle='--',
                linewidths=0.1,
                )
            fibo_title = ' fibo-'+fibo_data['fibo_type'][0:2]
            kwargs['hlines']=tpsl_lines
            kwargs['alines']=minmax_lines

        myrcparams = {'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':8}
        mystyle = mpf.make_mpf_style(base_mpf_style='charles',rc=myrcparams)

        filename = f"./plots/order_{symbol}.png"
        fig, axlist = mpf.plot(
            data,
            volume=True,volume_panel=1,
            **kwargs,
            type="candle",
            xrotation=0,
            ylabel='Price',
            style=mystyle,
            returnfig=True,
        )
        # print(axlist)
        ax1,*_ = axlist

        title = ax1.set_title(f'{symbol} {pd} ({config.timeframe} @ {data.index[-1]}{fibo_title})')
        title.set_fontsize(14)

        if showFibo:
            difference = fibo_data['difference']
            fibo_levels = fibo_data['fibo_levels']
            for idx, fibo_val in enumerate(fibo_data['fibo_values']):
                ax1.text(0,fibo_levels[idx] + difference * 0.02,f'{fibo_val}({fibo_levels[idx]})',fontsize=8,color=fibo_colors[idx],horizontalalignment='left')

            fibo_tp = fibo_data['tp']
            fibo_tp_txt = fibo_data['tp_txt']
            ax1.text(CANDLE_PLOT,fibo_tp - difference * 0.04,fibo_tp_txt,fontsize=8,color='g',horizontalalignment='right')
            fibo_sl = fibo_data['sl']
            fibo_sl_txt = fibo_data['sl_txt']
            ax1.text(CANDLE_PLOT,fibo_sl - difference * 0.04,fibo_sl_txt,fontsize=8,color='r',horizontalalignment='right')

        fig.savefig(filename)

        plt.close(fig)

        notify.Send_Image(msg, image_path=filename)
        # await sleep(2)
        if config.remove_plot:
            os.remove(filename)

    except Exception as ex:
        print(type(ex).__name__, symbol, str(ex))
        logger.exception(f'line_chart {symbol}')

    return

def line_notify_err(message):
    global is_send_notify_error, last_error_message
    is_send_notify_error = config.is_notify_api_error and is_send_notify_error
    if is_send_notify_error:
        line_notify(message)
        is_send_notify_error = False
        last_error_message = ''
    else:
        last_error_message = message
def line_notify_last_err():
    global is_send_notify_error, last_error_message
    if len(last_error_message):
        line_notify(last_error_message)
    is_send_notify_error = True
    last_error_message = ''

def line_notify(message):
    try:
        log_message = message.replace('\n', ',')
        logger.info(f'{log_message}')
        notify.Send_Text(message)
    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception(f'line_notify')

    return

def add_indicator(symbol, bars):
    df = pd.DataFrame(
        bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).map(
        lambda x: x.tz_convert("Asia/Bangkok")
    )
    df = df.set_index("timestamp")

    # เอาข้อมูลใหม่ไปต่อท้าย ข้อมูลที่มีอยู่
    if symbol in all_candles.keys() and len(df) < CANDLE_LIMIT:
        df = pd.concat([all_candles[symbol], df], ignore_index=False)

        # เอาแท่งซ้ำออก เหลืออันใหม่สุด
        df = df[~df.index.duplicated(keep='last')].tail(CANDLE_LIMIT)

    df = df.tail(CANDLE_LIMIT)

    if len(df) < CANDLE_SAVE:
        # print(f'less candles ({len(df)}/{CANDLE_SAVE}) for {symbol}, skip add_indicator')
        return df

    # คำนวนค่าต่างๆใหม่
    df['fast'] = 0
    df['mid'] = 0
    df['slow'] = 0
    df['MACD'] = 0
    df['MACDs'] = 0
    df['MACDh'] = 0
    df['ADX'] = 0
    df["RSI"] = 0
    df['STOCHk'] = 0
    df['STOCHd'] = 0

    try:
        fastType = config.fast_type 
        fastValue = config.fast_value
        midType = config.mid_type 
        midValue = config.mid_value        
        slowType = config.slow_type 
        slowValue = config.slow_value
        ADXPeriod = config.ADX_PERIOD

        if symbol in symbols_setting.index:
            # print(symbols_setting.loc[symbol])
            fastType = symbols_setting.loc[symbol]['fast_type']
            fastValue = int(symbols_setting.loc[symbol]['fast_value'])
            midType = symbols_setting.loc[symbol]['mid_type']
            midValue = int(symbols_setting.loc[symbol]['mid_value'])
            slowType = symbols_setting.loc[symbol]['slow_type']
            slowValue = int(symbols_setting.loc[symbol]['slow_value'])
            ADXPeriod = int(symbols_setting.loc[symbol]['adx_period'])

        if fastType == 'EMA':
            df['fast'] = ta.ema(df['close'],fastValue)
        elif fastType == 'SMA':
            df['fast'] = ta.sma(df['close'],fastValue)
        elif fastType == 'HMA':
            df['fast'] = ta.hma(df['close'],fastValue)
        elif fastType == 'RMA':
            df['fast'] = ta.rma(df['close'],fastValue)
        elif fastType == 'WMA':
            df['fast'] = ta.wma(df['close'],fastValue)
        elif fastType == 'VWMA':
            df['fast'] = ta.vwma(df['close'],df['volume'],fastValue)

        if midType == 'EMA':
            df['mid'] = ta.ema(df['close'],midValue)
        elif midType == 'SMA':
            df['mid'] = ta.sma(df['close'],midValue)
        elif midType == 'HMA':
            df['mid'] = ta.hma(df['close'],midValue)
        elif midType == 'RMA':
            df['mid'] = ta.rma(df['close'],midValue)
        elif midType == 'WMA':
            df['mid'] = ta.wma(df['close'],midValue)
        elif midType == 'VWMA':
            df['mid'] = ta.vwma(df['close'],df['volume'],midValue)

        if slowType == 'EMA':
            df['slow'] = ta.ema(df['close'],slowValue)
        elif slowType == 'SMA':
            df['slow'] = ta.sma(df['close'],slowValue)
        elif slowType == 'HMA':
            df['slow'] = ta.hma(df['close'],slowValue)
        elif slowType == 'RMA':
            df['slow'] = ta.rma(df['close'],slowValue)
        elif slowType == 'WMA':
            df['slow'] = ta.wma(df['close'],slowValue)
        elif slowType == 'VWMA':
            df['slow'] = ta.vwma(df['close'],df['volume'],slowValue)

        # cal MACD
        ewm_fast     = df['close'].ewm(span=config.MACD_FAST, adjust=False).mean()
        ewm_slow     = df['close'].ewm(span=config.MACD_SLOW, adjust=False).mean()
        df['MACD']   = ewm_fast - ewm_slow
        df['MACDs']  = df['MACD'].ewm(span=config.MACD_SIGNAL).mean()
        df['MACDh']  = df['MACD'] - df['MACDs']

        # cal ADX
        adx = ta.adx(df['high'],df['low'],df['close'],ADXPeriod)
        df['ADX']= adx[f'ADX_{ADXPeriod}']

        # cal RSI
        # change = df['close'].diff(1)
        # gain = change.mask(change<0,0)
        # loss = change.mask(change>0,0)
        # avg_gain = gain.ewm(com = config.RSI_PERIOD-1,min_periods=config.RSI_PERIOD).mean()
        # avg_loss = loss.ewm(com = config.RSI_PERIOD-1,min_periods=config.RSI_PERIOD).mean()
        # rs = abs(avg_gain / avg_loss)
        # df["RSI"] = 100 - ( 100 / ( 1 + rs ))
        df["RSI"] = ta.rsi(df['close'],config.RSI_PERIOD)

        # cal STO
        stoch_k = f'STOCHk_{config.STO_K_PERIOD}_{config.STO_D_PERIOD}_{config.STO_SMOOTH_K}'
        stoch_d = f'STOCHd_{config.STO_K_PERIOD}_{config.STO_D_PERIOD}_{config.STO_SMOOTH_K}'
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=config.STO_K_PERIOD, d=config.STO_D_PERIOD, smooth_k=config.STO_SMOOTH_K)
        df['STOCHk'] = stoch[stoch_k]
        df['STOCHd'] = stoch[stoch_d]

    except Exception as ex:
        print(type(ex).__name__, symbol, str(ex))
        logger.exception(f'add_indicator {symbol}')

    return df

def exchange_symbol(symbol):
    # return ':'.join([all_symbols[symbol]['symbol'],all_symbols[symbol]['quote']])
    return all_symbols[symbol]['symbol']
    # return symbol

"""
fetch_ohlcv - อ่านแท่งเทียน
exchange: binance exchange
symbol: coins symbol
timeframe: candle time frame
limit: จำนวนแท่งที่ต้องการ, ใส่ 0 หากต้องการให้เอาแท่งใหม่ที่ไม่มาครบ
timestamp: ระบุเวลาปัจจุบัน ถ้า limit=0
"""
async def fetch_ohlcv(exchange, symbol, timeframe, limit=1, timestamp=0):
    global all_candles
    ccxt_symbol = exchange_symbol(symbol)
    try:
        # # delay เพื่อให้กระจาย symbol ลด timeout
        # await sleep(randint(1,20))
        # กำหนดการอ่านแท่งเทียนแบบไม่ระบุจำนวน
        if limit == 0 and symbol in all_candles.keys():
            timeframe_secs = TIMEFRAME_SECONDS[timeframe]
            last_candle_time = int(pd.Timestamp(all_candles[symbol].index[-1]).tz_convert('UTC').timestamp())
            # ให้อ่านแท่งสำรองเพิ่มอีก 2 แท่ง
            limit = round(0.5+(timestamp-last_candle_time)/timeframe_secs)
        #     ohlcv_bars = await exchange.fetch_ohlcv(ccxt_symbol, timeframe, None, limit)
        # else:
        #     ohlcv_bars = await exchange.fetch_ohlcv(ccxt_symbol, timeframe, None, limit)
        async def fetch_ohlcv():
            return await exchange.fetch_ohlcv(ccxt_symbol, timeframe, None, limit)

        ohlcv_bars = await retry(fetch_ohlcv, limit=3)
        
        if len(ohlcv_bars):
            all_candles[symbol] = add_indicator(symbol, ohlcv_bars)
            # print(symbol, 'candles:', len(all_candles[symbol]))
        if symbol not in all_candles.keys():
            watch_list.remove(symbol)
            # print(f'{symbol} is no candlesticks, removed from watch_list')
            logger.debug(f'{symbol} is no candlesticks, removed from watch_list')
            # print(symbol)
    except Exception as ex:
        print(type(ex).__name__, symbol, str(ex))
        logger.exception(f'fetch_ohlcv {symbol}')
        line_notify_err(f'แจ้งปัญหาเหรียญ {symbol}:\nการอ่านแท่งเทียนผิดพลาด: {str(ex)}')
        if limit == 0 and symbol in all_candles.keys():
            print('----->', timestamp, last_candle_time, timestamp-last_candle_time, round(1.5+(timestamp-last_candle_time)/timeframe_secs))

async def fetch_ohlcv_trade(exchange, symbol, timeframe, limit=1, timestamp=0):
    await fetch_ohlcv(exchange, symbol, timeframe, limit, timestamp)
    await gather( go_trade(exchange, symbol) )

# order management zone --------------------------------------------------------
def new_order_history(symbol):
    global orders_history
    orders_history[symbol] = {
        # 'timestamp': 0,
        'positions': {}, 
        'orders': {},
        'orders_open': {}, 
        'win': 0, 
        'loss': 0, 
        'trade': 1,
        'last_loss': 0
    }
def open_order_history(symbol, positionSide:str, isTradeCount=True):
    global orders_history
    if symbol not in orders_history.keys():
        new_order_history(symbol)
    if positionSide not in orders_history[symbol]['positions'].keys() \
        or orders_history[symbol]['positions'][positionSide]['status'] != 'open':
        position = {}
        position['infos'] = {}
        position['status'] = 'open'
        orders_history[symbol]['positions'] = {}
        orders_history[symbol]['orders'] = {}
        orders_history[symbol]['orders_open'] = {}
        orders_history[symbol]['positions'][positionSide] = position
    if isTradeCount:
        orders_history[symbol]['trade'] = orders_history[symbol]['trade'] + 1
def close_order_history(symbol, positionSide:str):
    global orders_history
    if symbol not in orders_history.keys():
        new_order_history(symbol)
    if positionSide in orders_history[symbol]['positions'].keys():
        orders_history[symbol]['positions'][positionSide]['status'] = 'close'
    positions =  all_positions.loc[all_positions['symbol'] == symbol]
    if len(positions) == 0:
        return
    positionInfo = positions.iloc[-1]
    logger.debug(f'{symbol} close_order_history\n{positionInfo}')
    profit = 0
    if len(positions) > 0 and float(positionInfo["unrealizedProfit"]) != 0:
        profit = float(positionInfo["unrealizedProfit"])
    if profit > 0:
        orders_history[symbol]['win'] = orders_history[symbol]['win'] + 1
        orders_history[symbol]['last_loss'] = 0
    elif profit < 0:
        orders_history[symbol]['loss'] = orders_history[symbol]['loss'] + 1
        orders_history[symbol]['last_loss'] = orders_history[symbol]['last_loss'] + 1
    logger.debug(f'{symbol} win:{orders_history[symbol]["win"]} loss:{orders_history[symbol]["loss"]} trade:{orders_history[symbol]["trade"]} last_loss:{orders_history[symbol]["last_loss"]}')
def update_order_history(symbol, orderType:str, order, params={}):
    global orders_history
    if symbol not in orders_history.keys():
        new_order_history(symbol)
    try:
        positionSide = str(order['positionSide']).lower()
        if positionSide not in orders_history[symbol]['positions'].keys():
            position = {}
            position['infos'] = {}
            orders_history[symbol]['positions'][positionSide] = position
        elif 'infos' not in orders_history[symbol]['positions'][positionSide].keys():
            orders_history[symbol]['positions'][positionSide]['infos'] = {}

        position_infos = orders_history[symbol]['positions'][positionSide]['infos']
        position_info = position_infos[order['clientOrderId']] if order['clientOrderId'] in position_infos.keys() else {}
        if orderType.lower() == 'open':
            position_info['side'] = order['side']
            position_info['price'] = order['price']
            position_info['amount'] = order['amount']
            position_info['cost'] = order['cost']
            if 'lastPrice' in params.keys():
                position_info['last_price'] = params['lastPrice']
            orders_history[symbol]['positions'][positionSide]['status'] = 'open'
        elif orderType.lower() == 'tp':
            position_info['tp_price'] = params['stopPrice']
            # position_info['tp_amount'] = params['amount']
            position_info['tp_close_rate'] = params['closeRate']
        elif orderType.lower() == 'sl':
            position_info['sl_price'] = params['stopPrice']
            # position_info['sl_amount'] = params['amount']
        elif orderType.lower() == 'tl':
            position_info['tl_activatePrice'] = params['activatePrice']
            # position_info['tl_amount'] = params['amount']
            position_info['tl_callback'] = params['callback']
        elif orderType.lower() == 'close':
            position_info['close_price'] = order['price']
            position_info['close_amount'] = order['amount']
        orders_history[symbol]['positions'][positionSide]['infos'][order['clientOrderId']] = position_info
    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception(f'update_order_history')
        pass
async def update_open_orders(exchange, symbol):
    global orders_history
    try:
        total_cost = 0.0
        if symbol not in orders_history.keys():
            new_order_history(symbol)
        # # delay เพื่อให้กระจาย symbol ลด timeout
        # await sleep(randint(1,20))
        async def fetch_my_trades():
            return await exchange.fetch_my_trades(symbol)
        my_trades = await retry(fetch_my_trades, limit=3)
        columns = ['id', 'timestamp', 'datetime', 'symbol', 'order', 'type', 'side', 'price', 'amount', 'cost', 'fee', 'positionSide', 'clientOrderId']
        # display_columns = ['datetime', 'symbol', 'type', 'side', 'price', 'amount', 'cost', 'fee', 'clientOrderId']
        open_orders = pd.DataFrame(my_trades, columns=columns)
        # open_orders.drop(open_orders[~open_orders['clientOrderId'].str.startswith(bot_name.lower())].index, inplace=True)
        last_sell = open_orders.loc[open_orders['side'].eq('sell').idxmax()]
        if last_sell['side'] == 'sell':
            open_orders.drop(open_orders[~open_orders['timestamp'].gt(last_sell['timestamp'])].index, inplace=True)
            open_orders.reset_index(drop=True, inplace=True)
        if len(open_orders) == 0:
            return total_cost
        # logger.debug(f'{symbol} update_open_orders {open_orders}')
        for idx in range(0, len(open_orders)):
            order = open_orders.loc[idx]
            if order['side'] == 'buy' \
                and ( symbol not in orders_history.keys() \
                or 'spot' not in orders_history[symbol]['positions'].keys() \
                or order['clientOrderId'] not in orders_history[symbol]['positions']['spot']['infos'].keys() ):
                update_order_history(symbol, 'open', order, params={})
            elif order['clientOrderId'] in orders_history[symbol]['positions']['spot']['infos'].keys():
                orders_history[symbol]['positions']['spot']['status'] = 'open'
            # else:
            #     print(f'{symbol} is in orders')
            total_cost += order['cost']
        return total_cost
    except Exception as ex:
        print(type(ex).__name__, symbol, str(ex))
        logger.exception(f'update_open_orders {symbol}')
async def async_close_order_history(symbol, positionSide:str):
    close_order_history(symbol, positionSide)

def save_orders_history_csv(filename):
    oh_json = [{
        'symbol':symbol,
        'win':orders_history[symbol]['win'],
        'loss':orders_history[symbol]['loss'],
        'trade':orders_history[symbol]['trade']
    } for symbol in orders_history.keys()]
    oh_df = pd.DataFrame(oh_json)
    oh_df.to_csv(filename, index=False)
def save_orders_history_json(filename):
    with open(filename,"w", encoding='utf8') as json_file:
        json_string = json.dumps(orders_history, indent=2, ensure_ascii=False).encode('utf8')
        json_file.write(json_string.decode())
def load_orders_history_json(filename):
    global orders_history
    if os.path.exists(filename):
        with open(filename,"r", encoding='utf8') as json_file:
            orders_history = json.load(json_file)

# trading zone -----------------------------------------------------------------
def genClientOrderId(symbol, code, refCID=None):
    # global orders_history
    # order id len <= 32 chars
    # format: {botname}_{tf}_{timestamp}_{magic number}
    # sample: ema_3m_1674903982845_99999
    if refCID:
        clientId_tokens = str(refCID).split('_')
        tmst = int(clientId_tokens[2])
    else:
        tmst = int(round(datetime.now().timestamp()*1000))
    gen_order_id = f"{bot_name.lower()}_{code}_{tmst}_{config.magic_number}"
    gen_order_id = gen_order_id[0:32]
    # logger.debug(gen_order_id)
    return gen_order_id
async def spot_enter(exchange, symbol, amount, tf=config.timeframe):
    params={
        "client_id": genClientOrderId(symbol, tf),
    }
    ticker = await exchange.fetch_ticker(symbol)
    ask = float(ticker['ask'])
    order = await exchange.create_order(symbol, 'limit', 'buy', amount, price=ask, params=params)
    # print("Status : LONG ENTERING PROCESSING...")
    logger.debug(f'{symbol} spot_enter {str(order)}')
    open_order_history(symbol, 'spot')
    update_order_history(symbol, 'open', order, params={'lastPrice':ask})
    await sleep(1)
    return params['client_id']
#-------------------------------------------------------------------------------
async def spot_close(exchange, symbol, positionAmt, tf=config.timeframe, refCOID=None):
    params={
        "client_id": genClientOrderId(symbol, 'cl', refCOID),
    }
    ticker = await exchange.fetch_ticker(symbol)
    bid = float(ticker['bid'])
    order = await exchange.create_order(symbol, 'limit', 'sell', positionAmt, price=bid, params=params)
    logger.debug(f'{symbol} spot_close {str(order)}')
    close_order_history(symbol, 'spot')
    update_order_history(symbol, 'close', order)
    return params['client_id']
#-------------------------------------------------------------------------------
async def cancel_order(exchange, symbol, positionSide:str=None, refCOID=None):
    # try:
    #     await sleep(1)
    #     if positionSide == 'all':
    #         order = await exchange.cancel_all_orders(symbol, params={'conditionalOrdersOnly':False})
    #         logger.debug(f'{symbol} cancel_order {positionSide} {str(order)}')
    #     elif positionSide in ['long', 'short']:
    #         side = 'buy' if positionSide == 'short' else 'sell'
    #         open_orders = await exchange.fetch_open_orders(symbol)
    #         loops = [exchange.cancel_order(x['id'], x['symbol']) for x in open_orders if x['side'] == side]
    #         orders = await gather(*loops)
    #         logger.debug(f'{symbol} cancel_order {positionSide} {str(orders)}')
    # except Exception as ex:
    #     print(type(ex).__name__, symbol, str(ex))
    #     logger.exception(f'cancel_order {symbol}')
    return
#-------------------------------------------------------------------------------
async def spot_TPSL(exchange, symbol, amount, priceEntry, priceTP, priceSL, closeRate, refCOID=None):
    logger.debug(f'{symbol} spot_TPSL PriceEntry:{priceEntry}')
    await spot_TP(exchange, symbol, amount, priceTP, closeRate, refCOID)
    await spot_SL(exchange, symbol, amount, priceSL, refCOID)
    return
async def spot_TP(exchange, symbol, amount, priceTP, closeRate, refCOID=None):
    order = {}
    params = {}
    if refCOID:
        order['positionSide'] = 'spot'
        order['clientOrderId'] = refCOID
        params['stopPrice'] = priceTP
        params['amount'] = amount
        params['closeRate'] = closeRate
        update_order_history(symbol, 'tp', order, params)
        logger.debug(f'{symbol} spot_TP {str(order)} {str(params)}')
    return
async def spot_SL(exchange, symbol, amount, priceSL, refCOID=None):
    order = {}
    params = {}
    if refCOID:
        order['positionSide'] = 'spot'
        order['clientOrderId'] = refCOID
        params['stopPrice'] = priceSL
        params['amount'] = amount
        update_order_history(symbol, 'sl', order, params)
        logger.debug(f'{symbol} spot_SL {str(order)} {str(params)}')
    return
#-------------------------------------------------------------------------------
async def spot_TLSTOP(exchange, symbol, amount, priceTL, callbackRate, refCOID=None):
    order = {}
    params = {}
    activatePrice = priceTL
    if refCOID:
        order['positionSide'] = 'spot'
        order['clientOrderId'] = refCOID
        params['activatePrice'] = activatePrice
        params['amount'] = amount
        params['callback'] = callbackRate
        update_order_history(symbol, 'tl', order, params)
        logger.debug(f'{symbol} spot_TLSTOP {str(order)} {str(params)}')
    return activatePrice
#-------------------------------------------------------------------------------
async def cal_amount(exchange, symbol, leverage, costType, costAmount, closePrice, chkLastPrice):
    # คำนวนจำนวนเหรียญที่ใช้เปิดออเดอร์
    priceEntry = float(closePrice)
    # minAmount = float(all_symbols[symbol]['minAmount'])
    # minCost = float(all_symbols[symbol]['minCost'])
    if chkLastPrice:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            logger.debug(f'{symbol}:ticker\n{ticker}')
            priceEntry = float(ticker['last'])
        except Exception as ex:
            print(type(ex).__name__, str(ex))
    if costType=='#':
        amount = costAmount / priceEntry
    elif costType=='$':
        amount = costAmount * float(leverage) / priceEntry
    # elif costType=='M':
    #     # amount = priceEntry * minAmount / float(leverage) * 1.1
    #     amount =  minCost / float(leverage) / priceEntry * 1.1 
    else:
        # amount = (float(balance_entry)/100) * costAmount * float(leverage) / priceEntry
        amount = (float(balance_total)/100) * costAmount * float(leverage) / priceEntry

    p_amount = amount_to_precision(symbol, amount)
    p_priceAmt = amount_to_precision(symbol, amount * priceEntry / leverage)
    amount_precision = all_symbols[symbol]['amount_precision']

    logger.info(f'{symbol} lev:{leverage} close:{closePrice} last:{priceEntry} amt:{amount} p_amt:{p_amount} p:{amount_precision}')

    return (priceEntry, p_amount, p_priceAmt)

def crossover(tupleA, tupleB):
    return (tupleA[0] < tupleB[0] and tupleA[1] > tupleB[1])

def maomao(df, signalIdx):
    for i in range(config.back_days):
        last = df.iloc[signalIdx-i]
        last2nd = df.iloc[signalIdx-1-i]
        last3rd = df.iloc[signalIdx-2-i]
        # up
        # 1. แท่งเทียน​ อยู่เหนือ​ เส้น​ EMA 35 หรือ​ 32​ ก็ได้
        # 2. MACD > 0
        # 3. แท่งราคาปิด​ break ​แท่งเทียน​ ราคา ​High ก่อนหน้า
        if last['close'] > last['mid'] and \
            last['MACD'] > 0 and \
            last['close'] > last2nd['high'] and \
            last3rd['MACD'] < 0:
            return (True,False)
        # down
        # คิดตรงข้ามกับ up
        elif last['close'] < last['mid'] and \
            last['MACD'] < 0 and \
            last['close'] < last2nd['low'] and \
            last3rd['MACD'] > 0:
            return (False,True)
    return (False,False)

def cal_tpsl(symbol, amount, priceEntry, costAmount):
    cfg_tp = config.tp
    cfg_tp_close_rate = config.tp_close_rate
    cfg_sl = config.sl
    cfg_callback = config.callback
    cfg_active_tl = config.active_tl
    if symbol in symbols_setting.index:
        cfg_tp = float(symbols_setting.loc[symbol]['tp'])
        cfg_tp_close_rate = float(symbols_setting.loc[symbol]['tp_close_rate'])
        cfg_sl = float(symbols_setting.loc[symbol]['sl'])
        cfg_callback = float(symbols_setting.loc[symbol]['callback'])
        cfg_active_tl = float(symbols_setting.loc[symbol]['active_tl'])

    # คำนวน fibo
    if config.tp_pnl == 0 or config.sl_pnl == 0 or cfg_tp == 0 or cfg_sl == 0:
        if symbol in all_candles.keys() and len(all_candles[symbol]) >= CANDLE_SAVE:
            df = all_candles[symbol]
            lastPrice = df.iloc[-1]["close"]
            fibo_data = cal_minmax_fibo(symbol, df, lastPrice)
        else:
            return None

    if config.tp_pnl > 0:
        closeRate = config.tp_pnl_close_rate
        if config.is_percent_mode:
            priceTP = price_to_precision(symbol, priceEntry + (costAmount * (config.tp_pnl / 100.0) / amount))
        else:
            priceTP = price_to_precision(symbol, priceEntry + (config.tp_pnl / amount))
        if config.CB_AUTO_MODE == 1:
            callback_rate = cal_callback_rate(symbol, priceEntry, priceTP)
        if config.active_tl_pnl > 0:
            if config.is_percent_mode:
                priceTL = price_to_precision(symbol, priceEntry + (costAmount * (config.active_tl_pnl / 100.0) / amount))
            else:
                priceTL = price_to_precision(symbol, priceEntry + (config.active_tl_pnl / amount))
        cfg_callback = config.callback_pnl
    else:
        closeRate = cfg_tp_close_rate
        if cfg_tp > 0:
            priceTP = price_to_precision(symbol, priceEntry + (priceEntry * (cfg_tp / 100.0)))
        else:
            priceTP = fibo_data['tp']
        if cfg_active_tl > 0:
            priceTL = price_to_precision(symbol, priceEntry + (priceEntry * (cfg_active_tl / 100.0)))

    if config.sl_pnl > 0:
        if config.is_percent_mode:
            priceSL = price_to_precision(symbol, priceEntry - (costAmount * (config.sl_pnl / 100.0) / amount))
        else:
            priceSL = price_to_precision(symbol, priceEntry - (config.sl_pnl / amount))
        if config.CB_AUTO_MODE != 1:
            callback_rate = cal_callback_rate(symbol, priceEntry, priceSL)
    elif cfg_sl > 0:
        priceSL = price_to_precision(symbol, priceEntry - (priceEntry * (cfg_sl / 100.0)))
    else:
        priceSL = fibo_data['sl']

    if priceTL == 0.0:
        # RR = 1
        activationPrice = price_to_precision(symbol, priceEntry + abs(priceEntry - priceSL))
    else:
        activationPrice = priceTL

    if cfg_callback == 0.0:
        cfg_callback = callback_rate

    return (priceTP, priceSL, closeRate, activationPrice, cfg_callback)

async def go_trade(exchange, symbol, chkLastPrice=True):
    global all_positions, balance_entry, count_trade

    # delay เพื่อให้กระจายการ trade ของ symbol มากขึ้น
    delay = randint(5,9)
    # จัดลำดับการ trade symbol
    if symbol in orders_history.keys():
        winRate = orders_history[symbol]['win']/orders_history[symbol]['trade']
        if winRate > 0.5:
            delay = randint(1,2)
        elif winRate == 0.5:
             delay = randint(3,4)
    await sleep(delay)

    # อ่านข้อมูลแท่งเทียนที่เก็บไว้ใน all_candles
    if symbol in all_candles.keys() and len(all_candles[symbol]) >= CANDLE_SAVE:
        df = all_candles[symbol]
    else:
        # print(f'not found candles for {symbol} candlesticks')
        return
    # อ่านข้อมูล leverage ที่เก็บไว้ใน all_symbols
    if symbol in all_symbols.keys():
        leverage = all_symbols[symbol]['leverage']
    else:
        print(f'not found leverage for {symbol}')
        return

    marginType = all_symbols[symbol]['quote']

    hasSpotPosition = False
    positionAmt = 0.0

    positionInfo = all_positions.loc[all_positions['symbol']==symbol]

    if len(positionInfo) > 0 and float(positionInfo.iloc[-1]["total"]) != 0:
        positionAmt = float(positionInfo.iloc[-1]["total"])

    hasSpotPosition = (positionAmt > 0)

    # print(symbol, positionAmt, hasSpotPosition)

    try:
        signalIdx = config.signal_index
        tradeMode = config.trade_mode
        TPSLMode = config.tpsl_mode
        trailingStopMode = config.trailing_stop_mode
        costType = config.cost_type
        costAmount = config.cost_amount
        adxIn = config.adx_in
        positionLong = config.rsi_enter
        positionValueLong = config.rsi_enter_value
        exitLong = config.rsi_exit
        exitValueLong = config.rsi_exit_value
        stoValueLong = config.sto_enter
        stoValueShort = config.sto_exit
        if symbol in symbols_setting.index:
            signalIdx = int(symbols_setting.loc[symbol]['signal_index'])
            tradeMode = symbols_setting.loc[symbol]['trade_mode']
            TPSLMode = symbols_setting.loc[symbol]['tpsl_mode']
            trailingStopMode = symbols_setting.loc[symbol]['trailing_stop_mode']
            costType = symbols_setting.loc[symbol]['cost_type']
            costAmount = float(symbols_setting.loc[symbol]['cost_amount'])
            adxIn = int(symbols_setting.loc[symbol]['adx_in'])
            positionLong = symbols_setting.loc[symbol]['position_long']
            positionValueLong = int(symbols_setting.loc[symbol]['position_value_long'])
            exitLong = symbols_setting.loc[symbol]['exit_long']
            exitValueLong = int(symbols_setting.loc[symbol]['exit_value_long'])
            stoValueLong = int(symbols_setting.loc[symbol]['sto_enter_long'])
            stoValueShort = int(symbols_setting.loc[symbol]['sto_enter_short'])

        kwargs = dict()

        if config.strategy_mode == 'MAOMAO':
            (isSpotEnter, isSpotExit) = maomao(df, signalIdx)

        elif config.strategy_mode == 'EMA':
            fast = (df.iloc[signalIdx-1]['fast'], df.iloc[signalIdx]['fast'])
            mid = (df.iloc[signalIdx-1]['mid'], df.iloc[signalIdx]['mid'])
            slow = (df.iloc[signalIdx-1]['slow'], df.iloc[signalIdx]['slow'])
            
            isSpotEnter = crossover(fast, slow) # (fast[0] < slow[0] and fast[1] > slow[1])
            isSpotExit = crossover(mid, fast) # (fast[0] > mid[0] and fast[1] < mid[1])

            if config.confirm_macd_mode:
                isSpotEnter = isSpotEnter and (df.iloc[signalIdx][config.confirm_macd_by] > 0)

            if config.is_detect_sideway and isSpotEnter:
                sideways = detect_sideway_trend(df, config.atr_multiple, config.rolling_period, config.sideway_mode)
                if sideways[signalIdx] == 1:
                    isSpotEnter = False
                    print(f"[{symbol}] สถานะ : Sideway Tread skipping...")
                    logger.info(f'{symbol} -> Sideway Tread')
    
        elif config.strategy_mode == 'ADXRSI':
            rsi = (df.iloc[signalIdx-1]['RSI'], df.iloc[signalIdx]['RSI'])
            adxLast = df.iloc[signalIdx]['ADX']
            # close = (df.iloc[signalIdx-1]['close'], df.iloc[signalIdx]['close'])
            stoK = df.iloc[signalIdx]['STOCHk']
            stoD = df.iloc[signalIdx]['STOCHd']

            # logger.debug(f'{symbol} {rsi} {adxLast} {stoK} {stoD}')
            
            isSpotEnter = adxLast > adxIn and (
                (positionLong == 'up' and rsi[0] < positionValueLong and rsi[1] > positionValueLong) or
                (positionLong == 'down' and rsi[0] > positionValueLong and rsi[1] < positionValueLong)
                )
            isSpotExit = (exitLong == 'up' and rsi[1] > exitValueLong) or (exitLong == 'down' and rsi[1] < exitValueLong)
            
            isSTOSpotEnter = (stoK < stoValueLong and stoD < stoValueLong and stoK < stoD)

            if config.is_sto_mode:
                isSpotEnter = isSpotEnter and isSTOSpotEnter

            kwargs = dict(
                ADXIn=adxIn,
                RSIhi=positionValueLong,
                RSIlo=exitValueLong,
                STOhi=stoValueLong,
                STOlo=stoValueShort
            )
        else:
            isSpotEnter = False
            isSpotExit = False

        # print(symbol, isBullish, isBearish, fast, slow)

        closePrice = df.iloc[-1]["close"]

        if tradeMode == 'on' and isSpotExit == True and hasSpotPosition == True and config.sell_mode == 'on':
            count_trade = count_trade - 1 if count_trade > 0 else 0
            await spot_close(exchange, symbol, positionAmt)
            print(f"[{symbol}] สถานะ : {config.strategy_mode} Exit processing...")
            await cancel_order(exchange, symbol)
            # line_notify(f'{symbol}\nสถานะ : Long Exit')
            gather( line_chart(symbol, df, f'{symbol}\nสถานะ : {config.strategy_mode} Exit', f'{config.strategy_mode} EXIT', **kwargs) )

        notify_msg = []
        notify_msg.append(symbol)

        if isSpotEnter == True and hasSpotPosition == False:
            cfg_tp = config.tp
            cfg_tp_close_rate = config.tp_close_rate
            cfg_sl = config.sl
            cfg_callback = config.callback
            cfg_active_tl = config.active_tl
            if symbol in symbols_setting.index:
                cfg_tp = float(symbols_setting.loc[symbol]['tp'])
                cfg_tp_close_rate = float(symbols_setting.loc[symbol]['tp_close_rate'])
                cfg_sl = float(symbols_setting.loc[symbol]['sl'])
                cfg_callback = float(symbols_setting.loc[symbol]['callback'])
                cfg_active_tl = float(symbols_setting.loc[symbol]['active_tl'])

            print(f'{symbol:12} {config.strategy_mode}')
            fibo_data = cal_minmax_fibo(symbol, df, closePrice)
            if tradeMode == 'on' and balance_entry[marginType] > config.not_trade \
                and config.limit_trade > count_trade :
                count_trade = count_trade + 1
                (priceEntry, amount, priceAmt) = await cal_amount(exchange, symbol, leverage, costType, costAmount, closePrice, chkLastPrice)
                if amount <= 0.0:
                    print(f"[{symbol}] Status : {config.strategy_mode} NOT TRADE, Amount <= 0.0")
                else:
                    # ปรับปรุงค่า balance_entry
                    balance_entry[marginType] = balance_entry[marginType] - priceAmt
                    print('balance_entry', balance_entry[marginType])
                    refClientOrderId = await spot_enter(exchange, symbol, priceAmt)
                    print(f"[{symbol}] Status : {config.strategy_mode} ENTER PROCESSING...")
                    await cancel_order(exchange, symbol)
                    notify_msg.append(f'สถานะ : {config.strategy_mode}\nEnter\nราคา : {priceEntry}')

                    logger.debug(f'{symbol} {config.strategy_mode}\n{df.tail(3)}')
            
                    closeRate = 100.0
                    priceTL = 0.0
                    if TPSLMode == 'on':
                        notify_msg.append(f'# TPSL')
                        if config.tp_pnl > 0:
                            closeRate = config.tp_pnl_close_rate
                            if config.is_percent_mode:
                                priceTP = price_to_precision(symbol, priceEntry + (costAmount * (config.tp_pnl / 100.0) / amount))
                                fibo_data['tp_txt'] = f'TP PNL: {config.tp_pnl:.2f}% @{priceTP}'
                            else:
                                priceTP = price_to_precision(symbol, priceEntry + (config.tp_pnl / amount))
                                fibo_data['tp_txt'] = f'TP PNL: {config.tp_pnl:.2f}$ @{priceTP}'
                            fibo_data['tp'] = priceTP
                            if config.CB_AUTO_MODE == 1:
                                fibo_data['callback_rate'] = cal_callback_rate(symbol, priceEntry, priceTP)
                            if config.active_tl_pnl > 0:
                                if config.is_percent_mode:
                                    priceTL = price_to_precision(symbol, priceEntry + (costAmount * (config.active_tl_pnl / 100.0) / amount))
                                else:
                                    priceTL = price_to_precision(symbol, priceEntry + (config.active_tl_pnl / amount))
                            cfg_callback = config.callback_pnl
                        else:
                            closeRate = cfg_tp_close_rate
                            if cfg_tp > 0:
                                priceTP = price_to_precision(symbol, priceEntry + (priceEntry * (cfg_tp / 100.0)))
                                fibo_data['tp_txt'] = f'TP: {cfg_tp:.2f}% @{priceTP}'
                                fibo_data['tp'] = priceTP
                            else:
                                priceTP = fibo_data['tp']
                                fibo_data['tp_txt'] = f'TP: (AUTO) @{priceTP}'
                            if cfg_active_tl > 0:
                                priceTL = price_to_precision(symbol, priceEntry + (priceEntry * (cfg_active_tl / 100.0)))
                        notify_msg.append(fibo_data['tp_txt'])
                        notify_msg.append(f'TP close: {closeRate:.2f}%')
                        if config.sl_pnl > 0:
                            if config.is_percent_mode:
                                priceSL = price_to_precision(symbol, priceEntry - (costAmount * (config.sl_pnl / 100.0) / amount))
                                fibo_data['sl_txt'] = f'SL PNL: {config.sl_pnl:.2f}% @{priceSL}'
                            else:
                                priceSL = price_to_precision(symbol, priceEntry - (config.sl_pnl / amount))
                                fibo_data['sl_txt'] = f'SL PNL: {config.sl_pnl:.2f}$ @{priceSL}'
                            fibo_data['sl'] = priceSL
                            if config.CB_AUTO_MODE != 1:
                                fibo_data['callback_rate'] = cal_callback_rate(symbol, priceEntry, priceSL)
                        elif cfg_sl > 0:
                            priceSL = price_to_precision(symbol, priceEntry - (priceEntry * (cfg_sl / 100.0)))
                            fibo_data['sl_txt'] = f'SL: {cfg_sl:.2f}% @{priceSL}'
                            fibo_data['sl'] = priceSL
                        else:
                            priceSL = fibo_data['sl']
                            fibo_data['sl_txt'] = f'SL: (AUTO) @{priceSL}'
                        notify_msg.append(fibo_data['sl_txt'])

                        await spot_TPSL(exchange, symbol, amount, priceEntry, priceTP, priceSL, closeRate, refClientOrderId)
                        print(f'[{symbol}] Set TP {priceTP} SL {priceSL}')
                        
                    # if trailingStopMode == 'on' and closeRate < 100.0:
                    #     notify_msg.append('# TrailingStop')
                    #     if priceTL == 0.0:
                    #         # RR = 1
                    #         activationPrice = price_to_precision(symbol, priceEntry + abs(priceEntry - priceSL))
                    #     else:
                    #         activationPrice = priceTL

                    #     if cfg_callback == 0.0:
                    #         cfg_callback = fibo_data['callback_rate']
                    #         notify_msg.append(f'Call Back: (AUTO) {cfg_callback:.2f}%')
                    #     else:
                    #         notify_msg.append(f'Call Back: {cfg_callback:.2f}%')

                    #     activatePrice = await spot_TLSTOP(exchange, symbol, amount, activationPrice, cfg_callback, refClientOrderId)
                    #     print(f'[{symbol}] Set Trailing Stop {activationPrice:.4f}')
                    #     # callbackLong_str = ','.join(['{:.2f}%'.format(cb) for cb in callbackLong])

                    #     if priceTL == 0.0:
                    #         notify_msg.append(f'Active Price: (AUTO) @{activatePrice}')
                    #     elif config.tp_pnl > 0:
                    #         if config.is_percent_mode:
                    #             notify_msg.append(f'Active Price PNL: {config.active_tl_pnl:.2f}% @{activatePrice}')
                    #         else:
                    #             notify_msg.append(f'Active Price PNL: {config.active_tl_pnl:.2f}$ @{activatePrice}')
                    #     elif cfg_active_tl > 0:
                    #         notify_msg.append(f'Active Price: {cfg_active_tl:.2f}% @{activatePrice}')

                    gather( line_chart(symbol, df, '\n'.join(notify_msg), config.strategy_mode, fibo_data, **kwargs) )
                
            elif tradeMode != 'on' :
                fibo_data['tp_txt'] = 'TP'
                fibo_data['sl_txt'] = 'SL'
                gather( line_chart(symbol, df, f'{symbol}\nสถานะ : {config.strategy_mode}\nEnter', config.strategy_mode, fibo_data, **kwargs) )

    except Exception as ex:
        print(type(ex).__name__, symbol, str(ex))
        logger.exception(f'go_trade {symbol}')
        line_notify_err(f'แจ้งปัญหาเหรียญ {symbol}\nการเทรดผิดพลาด: {ex}')
        pass

async def load_all_symbols():
    global all_symbols, watch_list
    try:
        exchange = await getExchange()

        # t1=time.time()
        markets = await retry(exchange.fetch_markets, limit=3)

        # print(markets)
        mdf = pd.DataFrame(markets, columns=['id','quote','symbol'])
        mdf.drop(mdf[~mdf.quote.isin(config.margin_type)].index, inplace=True)
        # print(mdf.head())
        all_symbols = {r['id']:{
            'symbol':r['symbol'],
            'quote':r['quote'],
            'leverage':1,
            'amount_precision':2,
            'price_precision':8,
            } for r in mdf[['id','symbol','quote']].to_dict('records')}
        # print(all_symbols, len(all_symbols))
        # print(all_symbols.keys())
        if len(config.watch_list) > 0:
            watch_list_tmp = list(filter(lambda x: x in all_symbols.keys(), config.watch_list))
        else:
            watch_list_tmp = all_symbols.keys()
        # remove sysbol if in back_list
        watch_list = list(filter(lambda x: x not in config.back_list, watch_list_tmp))
        # print(watch_list)
        # print([(s, all_symbols[s]['symbol']) for s in all_symbols.keys() if s in watch_list])
        # t2=(time.time())-t1
        # print(f'ใช้เวลาหาว่ามีเหรียญ เทรดฟิวเจอร์ : {t2:0.2f} วินาที')
        
        print(f'total     : {len(all_symbols.keys())} symbols')
        print(f'target    : {len(watch_list)} symbols')

        logger.info(f'all:{len(all_symbols.keys())} watch:{len(watch_list)}')

    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception('load_all_symbols')

    finally:
        if exchange:
            await exchange.close()

async def fetch_first_ohlcv():
    try:
        exchange = await getExchange()

        # ครั้งแรกอ่าน 1000 แท่ง -> CANDLE_LIMIT
        limit = CANDLE_LIMIT

        if TIMEFRAME_SECONDS[config.timeframe] >= TIMEFRAME_SECONDS[config.START_TRADE_TF]:
            # อ่านแท่งเทียนแบบ async และ เทรดตามสัญญาน
            loops = [fetch_ohlcv_trade(exchange, symbol, config.timeframe, limit) for symbol in watch_list]
            await gather(*loops)
        else:
            # อ่านแท่งเทียนแบบ async แต่ ยังไม่เทรด
            loops = [fetch_ohlcv(exchange, symbol, config.timeframe, limit) for symbol in watch_list]
            await gather(*loops)

    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception('fetch_first_ohlcv')

    finally:
        if exchange:
            await exchange.close()

async def fetch_next_ohlcv(next_ticker):
    try:
        exchange = await getExchange()

        # กำหนด limit การอ่านแท่งเทียนแบบ 0=ไม่ระบุจำนวน, n=จำนวน n แท่ง
        limit = 0

        # อ่านแท่งเทียนแบบ async และ เทรดตามสัญญาน
        watch_list_rand = shuffle(watch_list.copy())
        loops = [fetch_ohlcv_trade(exchange, symbol, config.timeframe, limit, next_ticker) for symbol in watch_list]
        await gather(*loops)

    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception('fetch_next_ohlcv')

    finally:
        if exchange:
            await exchange.close()

async def mm_strategy():
    global all_positions, is_send_notify_risk, orders_history
    try:
        print('MM processing...')
        exchange = await getExchange()

        tickets = await retry(exchange.fetch_tickers, limit=3)
        balance = await retry(exchange.fetch_balance, limit=3)

        marginType = config.margin_type[0]

        ex_balances = balance['balances']
        mm_positions = [b for b in ex_balances if float(b['total']) != 0 
                    and b['asset'] != marginType
                    and f"{marginType}_{b['asset']}" in watch_list]

        # SL
        exit_loops = []
        cancel_loops = []
        mm_notify = []
        # exit all positions
        min_tl_rate = config.min_tl_rate / 100.0
        for position in mm_positions:
            symbol = f"{marginType}_{position['asset']}"
            marketPrice = tickets[symbol]['last'] if symbol in tickets.keys() else 0.0
            position_infos = orders_history[symbol]['positions']['spot']['infos']
            for coid in position_infos.keys():
                if 'sl_price' in position_infos[coid].keys():
                    if config.tpsl_mode == 'on' and position_infos[coid]['sl_price'] <= marketPrice:
                        print(f"[{symbol}] SL Exit {position_infos[coid]['sl_price']} > {marketPrice:.6f} (last)")
                        positionAmt = position_infos[coid]['amount']
                        await spot_close(exchange, symbol, positionAmt)
                        mm_notify.append(f'{symbol} : SL Exit')
                        # cancel_loops.append(cancel_order(exchange, symbol, 'long'))
                        logger.debug(f"[{symbol}] SL Exit {position_infos[coid]['sl_price']} > {marketPrice:.6f} (last)")
                    if config.tpsl_mode == 'on' and position_infos[coid]['tl_price'] >= marketPrice:
                        print(f"[{symbol}] TP Exit {position_infos[coid]['tl_price']} > {marketPrice:.6f} (last)")
                        positionAmt = position_infos[coid]['amount']
                        await spot_close(exchange, symbol, positionAmt)
                        mm_notify.append(f'{symbol} : TP Exit')
                        # cancel_loops.append(cancel_order(exchange, symbol, 'long'))
                        logger.debug(f"[{symbol}] TP Exit {position_infos[coid]['tl_price']} > {marketPrice:.6f} (last)")
                    
        try:
            if len(exit_loops) > 0:
                await gather(*exit_loops)
                hasMMPositions = True
        except Exception as ex:
            print(type(ex).__name__, str(ex))
            logger.exception('mm_strategy tpsl exit')

        if len(mm_notify) > 0:
            txt_notify = '\n'.join(mm_notify)
            line_notify(f'\nสถานะ...\n{txt_notify}')

#         hasMMPositions = False
#         balance = await exchange.fetch_balance()
#         if balance is None:
#             print('เกิดข้อผิดพลาดที่ api fetch_balance')
#             return
#         ex_positions = balance['info']['positions']
#         if len(ex_positions) == 0:
#             return
#         mm_positions = [position for position in ex_positions 
#             if position['symbol'] in all_symbols.keys() and
#                 all_symbols[position['symbol']]['quote'] in config.margin_type and 
#                 float(position['positionAmt']) != 0]

#         mm_positions = sorted(mm_positions, key=lambda k: float(k['unrealizedPrice']))

#         # sumProfit = sum([float(position['unrealizedPrice']) for position in mm_positions])
#         sumLongProfit = sum([float(position['unrealizedPrice']) for position in mm_positions if float(position['positionAmt']) >= 0])
#         sumShortProfit = sum([float(position['unrealizedPrice']) for position in mm_positions if float(position['positionAmt']) < 0])
#         sumProfit = sumLongProfit + sumShortProfit

#         sumLongMargin = sum([float(position['initialMargin']) for position in mm_positions if float(position['positionAmt']) >= 0])
#         sumShortMargin = sum([float(position['initialMargin']) for position in mm_positions if float(position['positionAmt']) < 0])
#         sumMargin = sumLongMargin + sumShortMargin

#         # count_trade = len(mm_positions)

#         # Money Management (MM) Strategy
#         logger.debug(f'MM Profit - Long[{sumLongProfit:.4f}] + Short[{sumShortProfit:.4f}] = All[{sumProfit:.4f}]')
#         # logger.debug(f'PNL: {config.TP_PNL}, {config.SL_PNL}')

#         cost_rate = 1.0
#         long_margin_rate = 1.0
#         short_margin_rate = 1.0
#         margin_rate = 1.0
#         if config.is_percent_mode:
#             cost_rate = config.cost_amount / 100.0
#             long_margin_rate = sumLongMargin / 100.0
#             short_margin_rate = sumShortMargin / 100.0
#             margin_rate = sumMargin / 100.0

#         tp_profit = config.TP_Profit * margin_rate
#         sl_profit = config.SL_Profit * margin_rate

#         logger.debug(f'MM TP/SL - All: {tp_profit:.4f}/-{sl_profit:.4f}')

#         # close all positions by TP/SL profit setting
#         if (tp_profit > 0 and sumProfit > tp_profit) or \
#             (sl_profit > 0 and sumProfit < -sl_profit):

#             exit_loops = []
#             cancel_loops = []
#             mm_notify = []
#             # exit all positions
#             for position in mm_positions:
#                 symbol = position['symbol']
#                 positionAmt = float(position['positionAmt'])
#                 if positionAmt > 0.0:
#                     print(f"[{symbol}] สถานะ : MM Long Exit processing...")
#                     exit_loops.append(spot_close(exchange, symbol, positionAmt))
#                     # line_notify(f'{symbol}\nสถานะ : MM Long Exit\nProfit = {sumProfit}')
#                     mm_notify.append(f'{symbol} : MM Long Exit')
#                     cancel_loops.append(cancel_order(exchange, symbol, 'long'))

#             try:
#                 if len(exit_loops) > 0:
#                     await gather(*exit_loops)
#                     hasMMPositions = True
#             except Exception as ex:
#                 print(type(ex).__name__, str(ex))
#                 logger.exception('mm_strategy exit all')

#             try:
#                 if len(cancel_loops) > 0:
#                     await gather(*cancel_loops)
#             except Exception as ex:
#                 print(type(ex).__name__, str(ex))
#                 logger.exception('mm_strategy cancel all')

#             if len(mm_notify) > 0:
#                 txt_notify = '\n'.join(mm_notify)
#                 line_notify(f'\nสถานะ...\n{txt_notify}\nProfit = {sumProfit:.4f}')
        
#         else:

#             # close target position by LONG/SHORT TP/SL PNL setting
#             exit_loops = []
#             cancel_loops = []
#             logger.debug(f'MM TP/SL PNL - Long: {config.TP_PNL_Long*cost_rate:.4f}/{-config.SL_PNL_Long*cost_rate:.4f} Short: {config.TP_PNL_Short*cost_rate:.4f}/{-config.SL_PNL_Long*cost_rate:.4f}')
#             if config.TP_PNL_Long > 0:
#                 tp_lists = [position for position in mm_positions if 
#                     float(position['positionAmt']) > 0.0 and 
#                     float(position['unrealizedPrice']) > config.TP_PNL_Long*cost_rate]
#                 if len(tp_lists) > 0:
#                     logger.debug(f'TP_PNL_Long {tp_lists}')
#                 for position in tp_lists:
#                     symbol = position['symbol']
#                     positionAmt = float(position['positionAmt'])
#                     unrealizedPrice = float(position['unrealizedPrice'])
#                     print(f"[{symbol}] สถานะ : MM Long Exit processing...")
#                     exit_loops.append(spot_close(exchange, symbol, positionAmt))
#                     line_notify(f'{symbol}\nสถานะ : MM Long Exit\nPNL = {unrealizedPrice}')
#                     cancel_loops.append(cancel_order(exchange, symbol, 'long'))
#             if config.SL_PNL_Long > 0:
#                 sl_lists = [position for position in mm_positions if 
#                     float(position['positionAmt']) > 0.0 and 
#                     float(position['unrealizedPrice']) < -config.SL_PNL_Long*cost_rate]
#                 if len(sl_lists) > 0:
#                     logger.debug(f'SL_PNL_Long {sl_lists}')
#                 for position in sl_lists:
#                     symbol = position['symbol']
#                     positionAmt = float(position['positionAmt'])
#                     unrealizedPrice = float(position['unrealizedPrice'])
#                     print(f"[{symbol}] สถานะ : MM Long Exit processing...")
#                     exit_loops.append(spot_close(exchange, symbol, positionAmt))
#                     line_notify(f'{symbol}\nสถานะ : MM Long Exit\nPNL = {unrealizedPrice}')
#                     cancel_loops.append(cancel_order(exchange, symbol, 'long'))

#             try:
#                 if len(exit_loops) > 0:
#                     await gather(*exit_loops)
#                     hasMMPositions = True
#             except Exception as ex:
#                 print(type(ex).__name__, str(ex))
#                 logger.exception('mm_strategy exit pnl')
#             try:
#                 if len(cancel_loops) > 0:
#                     await gather(*cancel_loops)
#             except Exception as ex:
#                 print(type(ex).__name__, str(ex))
#                 logger.exception('mm_strategy cancel pnl')

#         if hasMMPositions == False:
#             # notify risk
#             # balance = await exchange.fetch_balance()
#             # if balance is None:
#             #     print('เกิดข้อผิดพลาดที่ api fetch_balance')
#             #     return
#             # ex_positions = balance['info']['positions']

#             for marginType in config.margin_type:
#                 marginAsset = [asset for asset in balance['info']['assets'] if asset['asset'] == marginType][0]
#                 availableBalance = float(marginAsset['availableBalance'])
#                 initialMargin = float(marginAsset['initialMargin'])
#                 maintMargin = float(marginAsset['maintMargin'])
#                 totalRisk = abs(maintMargin) / (availableBalance+initialMargin) * 100
#                 if is_send_notify_risk == False and (config.risk_limit > 0) and (totalRisk > config.risk_limit):
#                     is_send_notify_risk = True
#                     logger.debug(f'MM {marginType} Risk Alert: {totalRisk:,.2f}% (limit {config.risk_limit:,.2f}%)')
#                     line_notify(f'แจ้งเตือน\n{marginType} Risk Alert: {totalRisk:,.2f}% (limit {config.risk_limit:,.2f}%)')
#                 elif totalRisk < config.risk_limit - 10.0:
#                     is_send_notify_risk = False
        
#         # clear margin
#         exit_loops = []
#         cancel_loops = []
#         mm_notify = []
#         # exit all positions
#         for position in mm_positions:
#             symbol = position['symbol']
#             initialMargin = float(position['initialMargin'])
#             if initialMargin <= config.Clear_Magin:
#                 print('remove', symbol, initialMargin)
#                 positionAmt = float(position['positionAmt'])
#                 if positionAmt > 0.0:
#                     print(f"[{symbol}] สถานะ : MM Long Exit processing...")
#                     exit_loops.append(spot_close(exchange, symbol, positionAmt))
#                     # line_notify(f'{symbol}\nสถานะ : MM Long Exit\nProfit = {sumProfit}')
#                     mm_notify.append(f'{symbol} : MM Long Remove')
#                     cancel_loops.append(cancel_order(exchange, symbol, 'long'))

#         try:
#             if len(exit_loops) > 0:
#                 await gather(*exit_loops)
#         except Exception as ex:
#             print(type(ex).__name__, str(ex))
#             logger.exception('mm_strategy clear exit')

#         try:
#             if len(cancel_loops) > 0:
#                 await gather(*cancel_loops)
#         except Exception as ex:
#             print(type(ex).__name__, str(ex))
#             logger.exception('mm_strategy clear cancel')

#         if len(mm_notify) > 0:
#             txt_notify = '\n'.join(mm_notify)
#             line_notify(f'\nสถานะ: Margin <= {config.Clear_Magin}\n{txt_notify}')

#         #loss counter
#         if config.Loss_Limit > 0:
#             for symbol in orders_history.keys():
#                 if orders_history[symbol]['last_loss'] >= config.Loss_Limit and symbol in watch_list:
#                     watch_list.remove(symbol)
#                     print(f'{symbol} removed from watch_list, last loss = {orders_history[symbol]["last_loss"]}')
#                     logger.info(f'{symbol} removed from watch_list, last loss = {orders_history[symbol]["last_loss"]}')

    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception('mm_strategy')
        line_notify_err(f'แจ้งปัญหาระบบ mm\nข้อผิดพลาด: {str(ex)}')

    finally:
        if exchange:
            await exchange.close()

async def update_tailing_stop():
    global orders_history
    try:
        print('TL Stop updating...')
        exchange = await getExchange()

        balance = await retry(exchange.fetch_balance, limit=3)

        marginType = config.margin_type[0]

        ex_balances = balance['balances']
        tl_positions = [b for b in ex_balances if float(b['total']) != 0 
                    and b['asset'] != marginType
                    and f"{marginType}_{b['asset']}" in watch_list]
        
        tl_notify = []
        min_tl_rate = config.min_tl_rate / 100.0
        for position in tl_positions:
            symbol = f"{marginType}_{position['asset']}"
            highPrice = all_candles[symbol]['high'][-1] if symbol in all_candles.keys() else 0.0
            closePrice = all_candles[symbol]['close'][-1] if symbol in all_candles.keys() else 0.0
            if highPrice <= 0.0 or closePrice <= 0.0:
                continue # skip if candle is not ready
            logger.debug(f"[{symbol}] TL high:{highPrice:.6f}, close:{closePrice:.6f}")
            if 'spot' not in  orders_history[symbol]['positions'].keys() \
                or 'infos' not in orders_history[symbol]['positions']['spot'].keys() \
                or orders_history[symbol]['positions']['spot']['status'] == 'close':
                # skip if position is closed
                continue
            position_infos = orders_history[symbol]['positions']['spot']['infos']
            for coid in position_infos.keys():
                if 'sl_price' in position_infos[coid].keys():
                    # calculate new SL form last candle high price
                    if position_infos[coid]['sl_price'] <= highPrice:
                        last_price = position_infos[coid]['last_price']
                        sl_percent = abs(last_price - position_infos[coid]['sl_price']) / last_price
                        if sl_percent < min_tl_rate:
                            sl_percent = min_tl_rate
                        new_sl = highPrice * (1.0 - sl_percent)
                        if position_infos[coid]['sl_price'] < new_sl:
                            print(f"[{symbol}] SL {position_infos[coid]['sl_price']} <= {highPrice:.6f}, {sl_percent:.4f}%, new SL:{new_sl:.8f}")
                            position_infos[coid]['last_price'] = highPrice
                            position_infos[coid]['sl_price'] = price_to_precision(symbol, new_sl)
                        # else:
                        #     print(f"[{symbol}] SL {position_infos[coid]['sl_price']} <= {marketPrice:.6f}, {sl_percent:.4f}%")
                        logger.debug(f"[{symbol}] SL {position_infos[coid]['sl_price']}, High {highPrice:.6f}, SL {sl_percent:.4f}%, new SL:{new_sl:.8f}")
                    elif config.tpsl_mode == 'on' and position_infos[coid]['sl_price'] <= closePrice:
                        print(f"[{symbol}] SL Exit {position_infos[coid]['sl_price']} > {closePrice:.6f} (close)")
                        positionAmt = position_infos[coid]['amount']
                        await spot_close(exchange, symbol, positionAmt)
                        tl_notify.append(f'{symbol} : SL Exit')
                        # cancel_loops.append(cancel_order(exchange, symbol, 'long'))
                        logger.debug(f"[{symbol}] SL Exit {position_infos[coid]['sl_price']} > {closePrice:.6f} (close)")
                else:
                    cfg_sl = config.sl
                    if symbol in symbols_setting.index:
                        cfg_sl = float(symbols_setting.loc[symbol]['sl'])
                    print(position_infos[coid])
                    priceEntry = position_infos[coid]['price']
                    costAmount = position_infos[coid]['cost']
                    amount = position_infos[coid]['amount']
                    if config.sl_pnl > 0:
                        if config.is_percent_mode:
                            new_sl = price_to_precision(symbol, priceEntry - (costAmount * (config.sl_pnl / 100.0) / amount))
                        else:
                            new_sl = price_to_precision(symbol, priceEntry - (config.sl_pnl / amount))
                    elif cfg_sl > 0:
                        new_sl = price_to_precision(symbol, priceEntry - (priceEntry * (cfg_sl / 100.0)))
                    else:
                        if symbol in all_candles.keys() and len(all_candles[symbol]) >= CANDLE_SAVE:
                            df = all_candles[symbol]
                            lastPrice = df.iloc[-1]["close"]
                            fibo_data = cal_minmax_fibo(symbol, df, lastPrice)
                            new_sl = fibo_data['sl']
                        else:
                            new_sl = price_to_precision(symbol, highPrice * (1.0 - min_tl_rate))
                    print(f"[{symbol}] New SL:{new_sl:.8f}")
                    position_infos[coid]['last_price'] = highPrice
                    position_infos[coid]['sl_price'] = price_to_precision(symbol, new_sl)
                    logger.debug(f'[{symbol}] New SL:{new_sl:.8f}')

        if len(tl_notify) > 0:
            txt_notify = '\n'.join(tl_notify)
            line_notify(f'\nสถานะ...\n{txt_notify}')

    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception('update_tailing_stop')
        line_notify_err(f'แจ้งปัญหาระบบ tl\nข้อผิดพลาด: {str(ex)}')

    finally:
        if exchange:
            await exchange.close()

    return

async def update_all_balance(notifyLine=False, updateOrder=False):
    global all_positions, balance_entry, balance_total, count_trade, orders_history
    try:
        exchange = await getExchange()

        tickets = await retry(exchange.fetch_tickers, limit=3)
        balance = await retry(exchange.fetch_balance, limit=3)
            
        marginType = config.margin_type[0]

        ex_balances = balance['balances']
        balances = [b for b in ex_balances if float(b['free']) != 0.0 
                    and b['asset'] != marginType
                    and f"{marginType}_{b['asset']}" in watch_list]
        # balances = [b for b in ex_balances if float(b['total']) != 0]

        all_positions = pd.DataFrame(balances, columns=BALANCE_COLUMNS)
        # all_positions.rename(columns={"asset": "symbol"}, inplace=True)
        all_positions["symbol"] = all_positions["asset"].apply(lambda x: f'{marginType}_{x}')
        # all_positions.reset_index(drop=True, inplace=True)
        # all_positions.index = all_positions.index + 1
        
        all_positions["marketPrice"] = all_positions['symbol'].apply(lambda x: tickets[x]['last'] 
                                                                     if x in tickets.keys() else 0.0)
        all_positions['unrealizedPrice'] = all_positions['total'].astype('float64') * all_positions["marketPrice"].astype('float64')
        all_positions['unrealizedPrice'] = all_positions['unrealizedPrice'].round(4)

        # all_positions["cost"] = all_positions['symbol'].apply(lambda x: await update_open_orders(exchange, x))
        # update open order
        if updateOrder:
            open_symbols = all_positions['symbol'].to_list()
            logger.debug(f'open_symbols: {open_symbols}')
            loops = [async_close_order_history(symbol, 'spot') for symbol in orders_history.keys() if symbol not in open_symbols]
            await gather(*loops)

            loops = [update_open_orders(exchange, symbol) for symbol in all_positions['symbol']]
            all_positions["Margin"] = await gather(*loops)
        else:
            def f(symbol):
                if 'spot' in orders_history[symbol]['positions'].keys():
                    if orders_history[symbol]['positions']['spot']['status'] == 'close':
                        return 0.0
                    infos = orders_history[symbol]['positions']['spot']['infos']
                    if len(infos.keys()) == 0:
                        return 0.0
                    else:
                        return sum(infos[k]['cost'] if 'cost' in infos[k].keys() else 0.0 for k in infos.keys())
                else:
                    return 0.0
            all_positions["Margin"] = all_positions['symbol'].apply(lambda x: f(x))

        all_positions["unrealizedProfit"] = all_positions['unrealizedPrice'] - all_positions["Margin"]
        all_positions['unrealizedProfit'] = all_positions['unrealizedProfit'].round(4)

        count_trade = len(all_positions)

        ub_msg = []
        ub_msg.append('รายงานสรุป')
        ub_msg.append(f'{bot_name} {bot_vesion}')

        if config.limit_trade > 0:
            ub_msg.append(f"# Count Trade: {count_trade}/{config.limit_trade}")
            print(f"Count Trade : {count_trade}/{config.limit_trade}")

        margin_balances = [b for b in ex_balances if b['asset'] == marginType]
        balance_entry[marginType] = float(margin_balances[0]['free'])

        sumUnrealizedPrice =  all_positions['unrealizedPrice'].astype('float64').sum()
        sumUnrealizedProfit =  all_positions['unrealizedProfit'].astype('float64').sum()

        balance_total = balance_entry[marginType] + sumUnrealizedPrice

        if len(all_positions) > 0:
            all_positions.sort_values(by=['unrealizedProfit'], ignore_index=True, ascending=False, inplace=True)
            all_positions.index = all_positions.index + 1
            print(all_positions[BALANCE_COLUMNS_DISPLAY])
        else:
            print('No Balances')

        ub_msg.append(f"# {marginType}\nFree: {balance_entry[marginType]:,.2f}/{config.not_trade:,.2f}\nPrice: {sumUnrealizedPrice:,.2f}\nProfit: {sumUnrealizedProfit:,.2f}")
        print(f"Balance === {marginType} Free: {balance_entry[marginType]:,.2f}/{config.not_trade:,.2f} Price: {sumUnrealizedPrice:,.2f} Profit: {sumUnrealizedProfit:,.2f}")
        balance_change = balance_total - start_balance_total if start_balance_total > 0 else 0
        ub_msg.append(f"# Total {balance_total:,.2f}\n# Change {balance_change:+,.2f}")
        print(f"Total ===== {balance_total:,.2f} Change: {balance_change:+,.2f}")

        if notifyLine:
            notify.Send_Text('\n'.join(ub_msg))

        keysList = list(orders_history.keys())
        logger.debug(f'symbol orders history: {keysList}')
        save_orders_history_csv(history_file_csv)
        save_orders_history_json(history_json_path)

    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception('update_all_balance')
        notify.Send_Text(f'แจ้งปัญหาระบบ update balance\nข้อผิดพลาด: {str(ex)}')
        pass

    finally:
        await exchange.close()

async def load_symbols_setting():
    global symbols_setting
    try:
        if config.CSV_NAME:
            symbols_setting = pd.read_csv(config.CSV_NAME, skipinitialspace=True)
            if any(x in CSV_COLUMNS for x in symbols_setting.columns.to_list()):
                symbols_setting.drop(symbols_setting[~symbols_setting.margin_type.isin(config.margin_type)].index, inplace=True)
                symbols_setting['id'] = symbols_setting['symbol']+symbols_setting['margin_type']
                symbols_setting.set_index('id', inplace=True)
                # เอาอันซ้ำออก เหลืออันใหม่สุด
                symbols_setting = symbols_setting[~symbols_setting.index.duplicated(keep='last')]

                # validate all values
                int_columns = [
                        'fast_value', 'mid_value', 'slow_value', 'signal_index', 'leverage'
                        ]
                float_columns = [
                        'cost_amount', 
                        'tp_long', 'tp_close_long', 'sl_long', 'callback_long', 'active_tl_long',
                        'tp_short', 'tp_close_short', 'sl_short', 'callback_short', 'active_tl_short'
                        ]
                symbols_setting[int_columns] = symbols_setting[int_columns].apply(pd.to_numeric, errors='coerce')
                symbols_setting[float_columns] = symbols_setting[float_columns].apply(pd.to_numeric, downcast='float', errors='coerce')
                symbols_setting.dropna(inplace=True)

                # print(symbols_setting.head())
                # print(symbols_setting.iloc[1])
                # validate all setting

                logger.info(f'success load symbols_setting from {config.CSV_NAME}')
            else:
                symbols_setting = pd.DataFrame(columns=CSV_COLUMNS)
                print(f'fail load symbols_setting from {config.CSV_NAME}, all columns not match')
                logger.info(f'fail load symbols_setting from {config.CSV_NAME}, all columns not match')

    except Exception as ex:
        symbols_setting = pd.DataFrame(columns=CSV_COLUMNS)
        print(type(ex).__name__, str(ex))
        logger.exception('load_symbols_setting')

# async def close_non_position_order(watch_list, positions_list):
#     try:
#         exchange = await getExchange()

#         loops = [cancel_order(exchange, symbol, 'all') for symbol in watch_list if symbol not in positions_list]
#         await gather(*loops)
    
#     except Exception as ex:
#         print(type(ex).__name__, str(ex))
#         logger.exception('update_all_balance')

#     finally:
#         if exchange:
#             await exchange.close()

async def main():
    global start_balance_total, is_send_notify_risk

    # pd.set_option('display.float_format', '{:.2f}'.format)

    marginList = ','.join(config.margin_type)
    if config.SANDBOX:
        bot_title = f'{bot_fullname} - {config.strategy_mode} - {config.timeframe} - {marginList} - (SANDBOX)'
    else:
        bot_title = f'{bot_fullname} - {config.strategy_mode} - {config.timeframe} - {marginList}'

    # set cursor At top, left (1,1)
    print(CLS_SCREEN+bot_title)

    await load_all_symbols()

    # await load_symbols_setting()

    load_orders_history_json(history_json_path)

    # แสดงค่า positions & balance
    await update_all_balance(notifyLine=config.summary_report, updateOrder=True)
    start_balance_total = balance_total

    # if config.IS_CLEAR_OLD_ORDER:
    #     await close_non_position_order(watch_list, all_positions['symbol'].to_list())
    # else:
    #     print(f'skip close_non_position_order')

    time_wait = TIMEFRAME_SECONDS[config.timeframe] # กำหนดเวลาต่อ 1 รอบ
    time_wait_ub = UB_TIMER_SECONDS[config.UB_TIMER_MODE] # กำหนดเวลา update balance
    if config.MM_TIMER_MIN == 0.0:
        time_wait_mm = time_wait_ub
    else:
        time_wait_mm = config.MM_TIMER_MIN*60

    # อ่านแท่งเทียนทุกเหรียญ
    t1=time.time()
    local_time = time.ctime(t1)
    print(f'get all candles: {local_time}')

    await fetch_first_ohlcv()
 
    t2=(time.time())-t1
    print(f'total time : {t2:0.2f} secs')
    logger.info(f'first ohlcv: {t2:0.2f} secs')

    await update_tailing_stop()

    try:
        start_ticker = time.time()
        next_ticker = start_ticker - (start_ticker % time_wait) # ตั้งรอบเวลา
        next_ticker += time_wait # กำหนดรอบเวลาถัดไป
        next_ticker_ub = start_ticker - (start_ticker % time_wait_ub)
        next_ticker_ub += time_wait_ub
        next_ticker_mm = start_ticker - (start_ticker % time_wait_mm)
        next_ticker_mm += time_wait_mm
        while True:
            seconds = time.time()

            if seconds >= next_ticker + TIME_SHIFT: # ครบรอบ
                # set cursor At top, left (1,1)
                print(CLS_SCREEN+bot_title)

                local_time = time.ctime(seconds)
                print(f'calculate new indicator: {local_time}')
                
                await update_all_balance(notifyLine=config.summary_report, updateOrder=True)

                t1=time.time()

                await fetch_next_ohlcv(next_ticker)

                t2=(time.time())-t1
                print(f'total time : {t2:0.2f} secs')
                logger.info(f'update ohlcv: {t2:0.2f} secs (include trade)')

                await update_tailing_stop()

                next_ticker += time_wait # กำหนดรอบเวลาถัดไป
                next_ticker_ub += time_wait_ub
                next_ticker_mm += time_wait_mm

                is_send_notify_risk = False
                line_notify_last_err()

                await sleep(10)

            else:
                # # mm strategy
                # if config.trade_mode == 'on' and seconds >= next_ticker_mm:
                #     # await mm_strategy()
                #     # await update_tailing_stop()
                #     next_ticker_mm += time_wait_mm
                #     line_notify_last_err()
                
                # display position
                if config.trade_mode == 'on' and seconds >= next_ticker_ub + TIME_SHIFT:
                    # set cursor At top, left (1,1)
                    print(CLS_SCREEN+bot_title)
                    balance_time = time.ctime(seconds)
                    print(f'last indicator: {local_time}, last balance: {balance_time}')
                    await update_all_balance()
                    next_ticker_ub += time_wait_ub
                    line_notify_last_err()

            await sleep(1)

    except KeyboardInterrupt:
        pass

    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception('main')


async def waiting():
    count = 0
    status = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    while True:
        await sleep(1)
        print('\r'+CGREEN+CBOLD+status[count%len(status)]+' waiting...\r'+CEND, end='')
        count += 1
        count = count%len(status)

if __name__ == "__main__":
    try:
        pathlib.Path('./plots').mkdir(parents=True, exist_ok=True)
        pathlib.Path('./logs').mkdir(parents=True, exist_ok=True)
        pathlib.Path('./datas').mkdir(parents=True, exist_ok=True)

        history_file_csv = './datas/orders_history.csv'
        history_json_path = './datas/orders_history.json'

        logger = logging.getLogger("App Log")
        logger.setLevel(config.LOG_LEVEL)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = RotatingFileHandler('./logs/app.log', maxBytes=250000, backupCount=10)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.info('start ==========')
        os.system("color") # enables ansi escape characters in terminal
        print(HIDE_CURSOR, end="")
        loop = get_event_loop()
        # แสดง status waiting ระหว่างที่รอ...
        loop.create_task(waiting())
        loop.run_until_complete(main())        

    except KeyboardInterrupt:
        print(CLS_LINE+'\rbye')

    except Exception as ex:
        print(type(ex).__name__, str(ex))
        logger.exception('app')
        line_notify(f'{bot_name} bot stop')

    finally:
        print(SHOW_CURSOR, end="")
        # save data
        # if os.path.exists(history_file_csv):
        #     os.rename(history_file_csv, history_file_csv.replace('.csv', f'{DATE_SUFFIX}.csv'))
        # save_orders_history_json(history_json_path)