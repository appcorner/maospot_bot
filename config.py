import configparser

def is_exist(group, name):
    return group in config.keys() and name in config[group].keys()

def get_list(group, name, default=[]):
    value = default
    try:
        if is_exist(group, name):
            value = [x.strip() for x in config[group][name].split(',')]
        else:
            print(f'config {group}.{name} not found, set default to {default}')
    except Exception as ex:
        print(type(ex).__name__, str(ex))
        print(f'config {group}.{name} not found, set default to {default}')
    return value

def get_list_float(group, name, default=[]):
    value = default
    try:
        if is_exist(group, name):
            value = [float(x.strip()) for x in config[group][name].split(',')]
        else:
            print(f'config {group}.{name} not found, set default to {default}')
    except Exception as ex:
        print(type(ex).__name__, str(ex))
        print(f'config {group}.{name} not found, set default to {default}')
    return value
    
def get_str(group, name, default=''):
    value = default
    try:
        if is_exist(group, name):
            value = config[group][name]
        else:
            print(f'config {group}.{name} not found, set default to {default}')
    except Exception as ex:
        print(type(ex).__name__, str(ex))
        print(f'config {group}.{name} not found, set default to {default}')
    return value

def get_int(group, name, default=0):
    value = default
    try:
        if is_exist(group, name):
            value = int(config[group][name])
        else:
            print(f'config {group}.{name} not found, set default to {default}')
    except Exception as ex:
        print(type(ex).__name__, str(ex))
        print(f'config {group}.{name} not found, set default to {default}')
    return value

def get_float(group, name, default=0.0):
    value = default
    try:
        if is_exist(group, name):
            value = float(config[group][name])
        else:
            print(f'config {group}.{name} not found, set default to {default}')
    except Exception as ex:
        print(type(ex).__name__, str(ex))
        print(f'config {group}.{name} not found, set default to {default}')
    return value


config = configparser.ConfigParser(interpolation=None)
config.optionxform = str
config_file = open("config.ini", mode='r', encoding='utf-8-sig')
config.readfp(config_file)

exchange = get_str('setting', 'exchange', 'bitkub')
#------------------------------------------------------------
# exchange
#------------------------------------------------------------
API_KEY = get_str(exchange,'api_key')
API_SECRET = get_str(exchange,'api_secret')
SANDBOX = (get_str(exchange,'sandbox', 'off') == 'on')

#------------------------------------------------------------
# line
#------------------------------------------------------------
LINE_NOTIFY_TOKEN = get_str('line','notify_token')
remove_plot = (get_str('line','remove_plot', 'off') == 'on')
summary_report = (get_str('line','summary_report', 'off') == 'on')
is_notify_api_error = (get_str('line','notify_api_error', 'off') == 'on')

#------------------------------------------------------------
# app_config
#------------------------------------------------------------
TIME_SHIFT = get_int('app_config', 'TIME_SHIFT', 5)
CANDLE_LIMIT = get_int('app_config', 'CANDLE_LIMIT', 1000)
CANDLE_PLOT = get_int('app_config', 'CANDLE_PLOT', 100)
LOG_LEVEL = get_int('app_config', 'LOG_LEVEL', 20)
UB_TIMER_MODE = get_int('app_config', 'UB_TIMER_MODE', 4)
if UB_TIMER_MODE < 0 or UB_TIMER_MODE > 5:
    UB_TIMER_MODE = 4
MM_TIMER_MIN = get_float('app_config', 'MM_TIMER_MIN', 0.0)
SWING_TF = get_int('app_config', 'SWING_TF', 5)
SWING_TEST = get_int('app_config', 'SWING_TEST', 2)
TP_FIBO = get_int('app_config', 'TP_FIBO', 2)
CB_AUTO_MODE = get_int('app_config', 'CB_AUTO_MODE', 1)
START_TRADE_TF = get_str('app_config', 'START_TRADE_TF', '4h')

#------------------------------------------------------------
# setting
#------------------------------------------------------------
timeframe = get_str('setting', 'timeframe', '5m')
magic_number = get_str('setting', 'magic_number', '12345')

signal_index = get_int('setting', 'signal_index', -2)
if signal_index > -1 or signal_index < -2:
    signal_index = -2

margin_type = get_list('setting', 'margin_type', ['THB'])

watch_list = get_list('setting', 'watch_list')
back_list = get_list('setting', 'back_list')

trade_mode = get_str('setting', 'trade_mode', 'off')
sell_mode = get_str('setting', 'sell_mode', 'on')

cost_type = get_str('setting', 'cost_type', '$')
cost_amount = get_float('setting', 'cost_amount', 1.5)

limit_trade = get_int('setting', 'limit_trade', 10)

not_trade = get_float('setting', 'not_trade', 10.0)

tpsl_mode = get_str('setting', 'tpsl_mode', 'on')
tp = get_float('setting', 'tp', 0.0)
sl = get_float('setting', 'sl', 0.0)
tp_close_rate = get_float('setting', 'tp_close_rate', 50.0)

trailing_stop_mode = get_str('setting', 'trailing_stop_mode', 'on')
callback = get_float('setting', 'callback', 0.0)
active_tl = get_float('setting', 'active_tl', 0.0)

strategy_mode = get_str('setting', 'strategy_mode', 'maomao').upper()

#------------------------------------------------------------
# indicator (share)
#------------------------------------------------------------
MID_TYPE = 'EMA'
MID_VALUE = 35

MACD_FAST = get_int('indicator', 'macd_fast')
MACD_SLOW = get_int('indicator', 'macd_slow')
MACD_SIGNAL = get_int('indicator', 'macd_signal')

ADX_PERIOD = get_int('indicator', 'adx_period', 14)
RSI_PERIOD = get_int('indicator', 'rsi_period', 14)

STO_K_PERIOD = get_int('indicator', 'sto_k_period', 14)
STO_SMOOTH_K = get_int('indicator', 'sto_smooth_k', 3)
STO_D_PERIOD = get_int('indicator', 'sto_d_period', 3)

#------------------------------------------------------------
# ema
#------------------------------------------------------------
FAST_TYPE = get_str('ema', 'fast_type').upper()
FAST_VALUE = get_int('ema', 'fast_value')
if strategy_mode == 'EMA':
    MID_TYPE = get_str('ema', 'mid_type').upper()
    MID_VALUE = get_int('ema', 'mid_value')
SLOW_TYPE = get_str('ema', 'slow_type').upper()
SLOW_VALUE = get_int('ema', 'slow_value')

confirm_macd_by = get_str('ema', 'confirm_macd_by', 'MACD')
confirm_macd_mode = get_str('ema', 'confirm_macd_mode', 'on') == 'on'
is_detect_sideway = get_str('ema', 'detect_sideway', 'on') == 'on'
sideway_mode = get_int('ema', 'sideway_mode', 2)
atr_multiple = get_float('ema', 'atr_multiple', 1.5)
rolling_period = get_int('ema', 'rolling_period', 15)

#------------------------------------------------------------
# adxrsi
#------------------------------------------------------------
adx_in = get_int('adxrsi', 'adx_in', 25)
rsi_enter = get_str('adxrsi', 'rsi_enter', 'up')
rsi_enter_value = get_int('adxrsi', 'rsi_enter_value', 70)
rsi_exit = get_str('adxrsi', 'rsi_exit', 'down')
rsi_exit_value = get_int('adxrsi', 'rsi_exit_value', 50)

is_sto_mode = get_str('adxrsi', 'sto_mode', 'on') == 'on'
sto_enter = get_int('adxrsi', 'sto_enter', 20)
sto_exit = get_int('adxrsi', 'sto_exit', 80)

#------------------------------------------------------------
# maomao
#------------------------------------------------------------
if strategy_mode == 'MAOMAO':
    MID_TYPE = 'EMA'
    MID_VALUE = get_int('maomao', 'ema_period', 35)
back_days = get_int('maomao', 'back_days', 3)

#------------------------------------------------------------
# symbols_setting
#------------------------------------------------------------
CSV_NAME = get_str('symbols_setting', 'csv_name', None)

#------------------------------------------------------------
# mm
#------------------------------------------------------------
is_percent_mode = get_str('mm', 'percent_mode', 'off') == 'on'

tp_pnl = get_float('mm', 'tp_pnl', 0.0)
sl_pnl = get_float('mm', 'sl_pnl', 0.0)
tp_pnl_close_rate = get_float('mm', 'tp_pnl_close_rate', 50.0)
active_tl_pnl = get_float('mm', 'active_tl_pnl', 0.0)
callback_pnl = get_float('mm', 'callback_pnl', 0.0)

tp_profit = get_float('mm', 'tp_profit', 0.0)
sl_profit = get_float('mm', 'sl_profit', 0.0)

clear_margin = get_float('mm', 'clear_margin', 0.01)

loss_limit = get_int('mm', 'loss_limit', 0)