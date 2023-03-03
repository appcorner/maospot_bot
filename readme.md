# stupid_bot

## mao spot bitkub
สามารถกำหนด strategy ที่ต้องการได้
- ema ใช้เส้นตัด slow mid fast
- maomao ใช้เส้น ema และ macd
- adxrsi ใช้ adx และ rsi และ/หรือ sto

## config.ini (rename จาก config.ini.sample)

    [bitkub]
    api_key = <bitkub api key>
    api_secret = <bitkub app secret>

    [line]
    notify_token = <line notify token>
    ;# กำหนด on/off เพื่อ เปิด/ปิด การลบรูปกราฟหลังจากใช้งานเสร็จ
    remove_plot = on
    ;# กำหนด on/off เพื่อ เปิด/ปิด การรายงานสรุป
    summary_report = on

    [app_config]
    TIME_SHIFT = 5

    ;# จำนวนแท่งเทียนที่ต้องการให้บอทใช้ทำงาน
    CANDLE_LIMIT = 200

    ;# จำนวนแท่งเทียนที่ต้องการแสดงกราฟ
    ;CANDLE_PLOT = 100

    ;# level การบันทึก log file ทั่วไปให้ใช้แบบ INFO
    ;# CRITICAL 50, ERROR 40, WARNING 30, INFO 20, DEBUG 10, NOTSET 0
    LOG_LEVEL = 10

    ;# กำหนดรอบเวลาในแสดง update balancec และ mm check
    ;# 0=timeframe, 1=15, 2=20, 3=30, 4=60, 5=timeframe/2 
    UB_TIMER_MODE = 3

    ;# กำหนดเาลาเป็น นาที ถ้าเป็น 0 จะใช้ UB_TIMER_MODE
    MM_TIMER_MIN = 0.5

    ;# จำนวน TF ในการตรวจหา swing low/high
    ;SWING_TF = 5

    ;# จำนวนค่า swing low/high ที่ใช้ในการคิด SL
    ;SWING_TEST = 2

    ;# level ของ fibo ที่ใช้ในการคิด TP
    ;TP_FIBO = 2

    ;# คำนวน callback rate จาก 1 = TP, 2 = SL
    ;CB_AUTO_MODE = 1

    ;# กำหนด timeframe ขั้นต่ำ ที่ต้องการเทรดเมื่อเริ่มการทำงานครั้งแรก (default = 4h)
    ;START_TRADE_TF = 1h

    [setting]
    ; 1m, 5m, 15m, 30m, 1h, 4h, 1d
    timeframe = 1h
    ; กำหนดสัญญานที่แท่ง -1 หรือ -2 เท่านั้น, default = -2
    signal_index = -2
    margin_type = THB

    ; ระบุ symbol ที่ต้องการใน watch_list, back_list
    ;watch_list = THB_BTC,THB_ETH,THB_KUB,THB_XRP,THB_USDT,THB_BNB,THB_ADA,THB_IOST,THB_DOGE,THB_BUSD,THB_DOT,THB_NEAR,THB_ALPHA,THB_CRV,THB_LUNC,THB_ALGO,THB_1INCH,THB_ATOM,THB_LDO,THB_STG,THB_APE,THB_AXL,THB_GALA,THB_IMX,THB_JFIN,THB_OP,THB_SAND,THB_SIX,THB_ZIL
    back_list = THB_LUNA2

    ;# กำหนด on/off ในการเทรด
    trade_mode = on
    sell_mode = on

    ; กำหนดรูปการคิด cost # $ %
    cost_type = $
    ; order ขั้นต่ำคือ 10 THB
    cost_amount = 50

    ; กำหนดจำนวน เหรียญที่ซื้อ ไม่เกิน limit_trade
    limit_trade = 10

    ; กำหนดจำนวน balance ขั้นต่ำ จะไม่เปิด order ใหม่ ถ้า balance เหลือต่ำกว่า not_trade
    not_trade = 50

    ; ห้ามเปิดใช้ เพราะยังไม่ทำงาน
    tpsl_mode = off
    ;tp_long = 10.0
    ;tp_short = 10.0
    ;tp_close_long = 50.0
    ;tp_close_short = 50.0
    ;sl_long = 4.0
    ;sl_short = 4.0

    ; ห้ามเปิดใช้ เพราะยังไม่ทำงาน
    trailing_stop_mode = off
    ;callback_long = 5.0
    ;callback_short = 5.0
    ;active_tl_long = 10.0
    ;active_tl_short = 10.0

    ; ระบุ type fast,slow => EMA, SMA, HMA, RMA, WMA, VWMA
    fast_type = EMA
    fast_value = 8
    mid_type = EMA
    mid_value = 32
    slow_type = EMA
    slow_value = 32

    ; สำหรับคำนวน macd, default คือค่ามาตราฐาน
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    ; สำหรับคำนวน rsi sto, default คือค่ามาตราฐาน
    rsi_period = 14
    sto_k_period = 14
    sto_smooth_k = 3
    sto_d_period = 3

    ; กำหนด strategy เป็น ema, maomao, adxrsi
    strategy_mode = adxrsi

    [ema]
    confirm_macd_mode = on
    detect_sideway = on
    sideway_mode = 2
    atr_multiple = 1.5
    rolling_period = 15

    [maomao]
    # 1. แท่งเทียน​ อยู่เหนือ​ เส้น​ EMA 35 หรือ​ 32​ ก็ได้
    # 2. MACD > 0
    # 3. แท่งราคาปิด​ break ​แท่งเทียน​ ราคา ​High ก่อนหน้า
    # EMA ในข้อ 1 ให้กำหนดค่าใน setting ดังนี้ mid_type = EMA, mid_value = 35
    # จำนวนวันย้อนหลัง สำหรับใช้ตรวจสัญญาน
    back_days = 3

    [adxrsi]
    ;# @อาร์ม setting
    adx_in = 5
    rsi_enter = down
    rsi_enter_value = 25
    rsi_exit = up
    rsi_exit_value = 80

    ;# STO on/off
    sto_mode = off
    sto_enter = 20
    sto_exit = 80

    ; (ยังใช่ไม่ได้) กำหนดค่า setting แยกตาม symbol
    [symbols_setting]
    ; ชื่อไฟล์ที่เก็บ setting ต้องเป็นไฟล์ csv
    csv_name = symbol_config.csv

    ; (mm ยังใช่ไม่ได้)
    [mm]
    ;# ต้องให้คำนวน TP/SL auto ให้กำหนดค่า tp_pnl_long, tp_pnl_short, sl_pnl_long, sl_pnl_short เป็น 0.0 
    ;# ค่าตัวแปรต่างๆ กำหนดค่าเป็น 0 หรือ comment ถ้าต้องการปิดการทำงาน

    ;# TP/SL แบบ PNL เพื่อปิด position โดยใช้ค่า PNL amount มาเป็นตัวกำหนด

    ;# ใส่ค่าเป็น amount (percent_mode=off) หรือ % (percent_mode=on) จะคำนวน amount ให้จาก % x cost_amount
    ;# ค่า close_rate ใส่เป็น % เท่านั้น
    ;# callback rate ranges from 0.1% to 5%, 0.0 for auto

    percent_mode = on

    ;# ส่วนนี้ใช้กับ long position
    tp_pnl_long = 0.30
    tp_pnl_close_rate_long = 25.0
    sl_pnl_long = 0.10
    ;# ค่า active tl ถ้ากำหนดเป็น 0.0 เพื่อให้บอทคำนวนค่าให้
    active_tl_pnl_long = 0.0
    callback_pnl_long = 5.0

    ;# ส่วนนี้ใช้กับ short position
    tp_pnl_short = 0.30
    tp_pnl_close_rate_short = 25.0
    sl_pnl_short = 0.10
    ;# ค่า active tl ถ้ากำหนดเป็น 0.0 เพื่อให้บอทคำนวนค่าให้
    active_tl_pnl_short = 0.0
    callback_pnl_short = 5.0

    ;# TP/SL Profit เพื่อปิด positions ทั้งหมด โดยใช้ค่าผลรวมของ profit มาเป็นตัวกำหนด 
    ;# จะทำจะงานตามรอบเวลาที่กำหนดไว้ที่ MM_TIMER_MIN
    
    ;# ส่วนนี้สำหรับคำนวนรวมทุก positions
    ;tp_profit = 1.5
    ;sl_profit = 1.4

    ;# ส่วนนี้สำหรับคำนวนแยก long position (ถ้ากำหนดแบบรวมไว้ ค่านี้จะไม่ถูกใช้)
    tp_profit_long = 3.0
    sl_profit_long = 1.5

    ;# ส่วนนี้สำหรับคำนวนแยก short position (ถ้ากำหนดแบบรวมไว้ ค่านี้จะไม่ถูกใช้)
    tp_profit_short = 3.0
    sl_profit_short = 1.5

    ;# กำหนดค่าการปิด position ที่มี margin น้อยกว่า clear_margin, default คีอ 0.01
    clear_margin = 0.01

    ;# ระบบจะนับ loss ถ้าเกิด loss เกิน loss_limit จะทำการเอาเหรียญออกจาก watch_list ชั่วคราว
    ;# เมื่อปิดเปิดบอทใหม่ watch_list จะเป็นค่าเดิมที่ตั้งไว้
    loss_limit = 0

## donate
- ETH: 0xeAfe7f1Db46E2c007b60174EB4527ab38bd54B54
- DOGE: DCkpCgt1HUVhCsvJVW7YA4E4NkxMsLHPz8