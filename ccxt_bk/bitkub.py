# -*- coding: utf-8 -*-

# PLEASE DO NOT EDIT THIS FILE, IT IS GENERATED AND WILL BE OVERWRITTEN:
# https://github.com/ccxt/ccxt/blob/master/CONTRIBUTING.md#how-to-contribute-code
# NOTE: this is modified version of bitkub from
#       https://github.com/binares/ccxt-unmerged/tree/master/ccxt_unmerged

from ccxt.base.exchange import Exchange
import json


class bitkub(Exchange):

    def describe(self):
        return self.deep_extend(super(bitkub, self).describe(), {
            'id': 'bitkub',
            'name': 'bitkub',
            'countries': ['TH'],
            'version': 'v1',
            'has': {
                'CORS': False,
                'spot': True,
                'future': False,
                'fetchCurrencies': False,
                'fetchOHLCV': True,
                'withdraw': False,
                'publicAPI': True,
                'privateAPI': True,
                'fetchMarkets': True,
                'fetchTicker': True,
                'fetchTickers': True,
                'fetchOrderBook': True,
                'fetchTrades': True,
                'fetchMyTrades': True,
                'fetchBalance': True,
                'createOrder': True,
                'cancelOrder': True,
                'fetchDepositAddress': True,
            },
            'timeframes': {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '30m': 1800,
                '1h': 3600,
                '4h': 14400,
                '1d': 86400,
            },
            'tf_map': { '1m': '1', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '4h': '240', '1d': '1D' },
            'urls': {
                'logo': 'https://www.bitkub.com/static/images/logo-white.png',
                'api': 'https://api.bitkub.com',
                'www': 'https://www.bitkub.com',
                'doc': 'https://github.com/bitkub/bitkub-official-api-docs',
                'fees': 'https://www.bitkub.com/fee/cryptocurrency',
            },
            'api': {
                'public': {
                    'get': [
                        'api/status',
                        'api/servertime',
                        'api/market/symbols',
                        'api/market/ticker',
                        'api/market/trades',
                        'api/market/bids',
                        'api/market/asks',
                        'api/market/books',
                        'api/market/depth',
                        'tradingview/history',
                    ],
                },
                'private': {
                    'post': [
                        'api/market/wallet',
                        'api/market/balances',
                        'api/market/v2/place-bid',
                        'api/market/v2/place-ask',
                        'api/market/place-ask-by-fiat',
                        'api/market/v2/cancel-order',
                        'api/market/my-open-orders',
                        'api/market/my-order-history',
                        'api/market/order-info',
                        'api/crypto/addresses',
                        'api/crypto/withdraw',
                        'api/crypto/deposit-history',
                        'api/crypto/withdraw-history',
                        'api/crypto/generate-address',
                        'api/fiat/accounts',
                        'api/fiat/withdraw',
                        'api/fiat/deposit-history',
                        'api/fiat/withdraw-history',
                        'api/market/wstoken',
                        'api/user/limits',
                        'api/user/trading-credits',
                    ],
                },
            },
            'timeout': 5000,
            'rateLimit': 1000,
            'precision': {
                'price': 2,
                'amount': 8,
                'cost': 2,
            },
            'fees': {
                'trading': {
                    'tierBased': False,
                    'percentage': True,
                    'taker': 0.0025,
                    'maker': 0.0025,
                },
            },
        })

    def fetch_markets(self, params={}):
        response = self.publicGetApiMarketSymbols(params)
        markets = response['result']
        settings = self.describe()
        result = []
        for i in range(0, len(markets)):
            market = markets[i]
            id = self.safe_string(market, 'symbol')
            currencySymbol = id.split('_')
            base = currencySymbol[1]
            quote = currencySymbol[0]
            baseId = base.lower()
            quoteId = quote.lower()
            symbol = base + '/' + quote
            result.append({
                'id': id,
                'symbol': symbol,
                'base': base,
                'quote': quote,
                'baseId': baseId,
                'quoteId': quoteId,
                "type": "spot",
                "spot": True,
                "future": False,
                'info': market,
                'active': True,
                'precision': settings['precision'],
                'limits': {
                    'amount': {
                        'min': None,
                        'max': None,
                    },
                    'price': {
                        'min': None,
                        'max': None,
                    },
                    'cost': {
                        'min': 10,
                        'max': None,
                    },
                },
            })
        return result

    def fetch_balance(self, params={}):
        self.load_markets()
        response = self.privatePostApiMarketBalances(params)
        markets = response['result']
        keyMarkets = list(markets.keys())
        result = {}
        free = {}
        result['info'] = markets
        result['balances'] = []
        for i in range(0, len(keyMarkets)):
            key = keyMarkets[i]
            market = markets[key]
            available = self.safe_float(market, 'available')
            reserved = self.safe_float(market, 'reserved')
            free[key] = available
            account = {
                'asset': key,
                'free': available,
                'locked': reserved,
                'total': available + reserved,
            }
            result['balances'].append(account)
        # return self.parse_balance(result)
        return result

    def fetch_order_book(self, symbol, limit=None, params={}):
        self.load_markets()
        if limit is None:
            limit = 10
        request = {
            'sym': self.market_id(symbol),
            'lmt': limit,
        }
        response = self.publicGetApiMarketBooks(self.extend(request, params))
        orderbook = response['result']
        lastBidTime = orderbook['bids'][0][1]
        lastAskTime = orderbook['asks'][0][1]
        timestamp = lastBidTime if (lastBidTime > lastAskTime) else lastAskTime
        return self.parse_order_book(orderbook, symbol, int(timestamp) * 1000, 'bids', 'asks', 3, 4)

    def parse_ticker(self, ticker, market=None):
        symbol_id = None
        symbol = None
        if market is not None:
            if isinstance(market, list):
                symbol_id = market[0]['id']
                symbol = market[0]['symbol']
            else:
                symbol_id = market['id']
                symbol = market['symbol']
        timestamp = self.milliseconds()
        last = self.safe_float(ticker, 'last')
        change = self.safe_float(ticker, 'change')
        open = None
        average = None
        if (last is not None) and (change is not None):
            open = last - change
            average = (last + open) / 2
        baseVolume = self.safe_float(ticker, 'baseVolume')
        quoteVolume = self.safe_float(ticker, 'quoteVolume')
        vwap = None
        if quoteVolume is not None:
            if (baseVolume is not None) and (baseVolume > 0):
                vwap = quoteVolume / baseVolume
        return {
            'id': symbol_id,
            'symbol': symbol,
            'timestamp': timestamp,
            'datetime': self.iso8601(timestamp),
            'high': self.safe_float(ticker, 'high24hr'),
            'low': self.safe_float(ticker, 'low24hr'),
            'bid': self.safe_float(ticker, 'highestBid'),
            'bidVolume': None,
            'ask': self.safe_float(ticker, 'lowestAsk'),
            'askVolume': None,
            'vwap': vwap,
            'open': open,
            'close': last,
            'last': last,
            'previousClose': self.safe_float(ticker, 'prevClose'),
            'change': change,
            'percentage': self.safe_float(ticker, 'percentChange'),
            'average': average,
            'baseVolume': baseVolume,
            'quoteVolume': quoteVolume,
            'info': ticker,
        }

    def fetch_ticker(self, symbol, params={}):
        self.load_markets()
        market = self.market(symbol)
        request = {
            'sym': market['id'],
        }
        response = self.publicGetApiMarketTicker(self.extend(request, params))
        return self.parse_ticker(self.safe_value(response, market['id']), market)

    def fetch_tickers(self, symbols=None, params={}):
        self.load_markets()
        response = self.publicGetApiMarketTicker(params)
        keys = list(response.keys())
        tickers = []
        for i in range(0, len(keys)):
            market = self.safe_value(self.markets_by_id, keys[i])
            tickers.append(self.parse_ticker(response[keys[i]], market))
        return self.filter_by_array(tickers, 'id', symbols)

    def fetch_ohlcv(self, symbol, timeframe='1m', since=None, limit=None, params={}):
        self.load_markets()
        market = self.market(symbol)
        sym = str(market['id']).split('_')
        request = {
            'symbol': f'{sym[1]}_{sym[0]}',
            'resolution': self.tf_map[timeframe],
        }
        duration = self.timeframes[timeframe]
        if limit is None:
            limit = 1
        timerange = duration * limit
        if since is None:
            request['to'] = int(self.milliseconds() / 1000)
            request['from'] = request['to'] - timerange
        else:
            request['from'] = int(since / 1000)
            request['to'] = self.sum(request['from'], timerange)
        ohlcv = self.publicGetTradingviewHistory(self.extend(request, params))
        result = []
        if 's' in ohlcv.keys() and ohlcv['s'] == 'ok':
            objOHLCV = list(ohlcv['c'] or [].values())
            length = len(objOHLCV)
            for i in range(0, length):
                ts = self.safe_timestamp(ohlcv['t'], i)
                open = float(ohlcv['o'][i])
                high = float(ohlcv['h'][i])
                low = float(ohlcv['l'][i])
                close = float(ohlcv['c'][i])
                vol = float(ohlcv['v'][i])
                # result.append([])
                newOHLCV = [ts, open, high, low, close, vol]
                result.append(newOHLCV)
        return result

    def create_order(self, symbol, type, side, amount, price=None, params={}):
        self.load_markets()
        market = self.market(symbol)
        if type == 'market' and price == None:
            price = 0
        request = {
            'sym': market['id'],
            'amt': self.price_to_precision(symbol, amount) if side == 'buy' else amount,
            'rat': self.amount_to_precision(symbol, price),
            'typ': type,
        }
        if 'sellbyfiat' in params.keys() and params['sellbyfiat'] == True:
            method = 'privatePostApiMarketV2PlaceBid' if side == 'buy' else 'privatePostApiMarketPlaceAskByFiat'
        else:
            method = 'privatePostApiMarketV2PlaceBid' if side == 'buy' else 'privatePostApiMarketV2PlaceAsk'
        response = getattr(self, method)(self.extend(request, params))
        if 'result' in response.keys():
            order = response['result']
            id = self.safe_string(order, 'id')
            r_price = float(order['rat']) if float(order['rat']) > 0 else price
            if side == 'buy':
                r_amount = float(order['rec']) if float(order['rec']) > 0 and price > 0 else amount/price
                r_cost = float(order['amt']) if float(order['amt']) > 0 else amount
            else:
                r_amount = float(order['amt']) if float(order['amt']) > 0 else amount
                r_cost = float(order['rec']) if float(order['rec']) > 0 and price > 0 else amount*price
            clientOrderId = self.safe_string(order, 'ci', '')
            return {
                'id': id,
                'positionSide': 'spot',
                'side': side,
                'price': r_price,
                'amount': r_amount,
                'cost': r_cost,
                'clientOrderId': clientOrderId,
                'info': order,
            }
        else:
            return response

    def school_round(self, a_in, n_in):
        ''' python uses "banking round; while this round 0.05 up '''
        if (a_in * 10 ** (n_in + 1)) % 10 == 5:
            return round(a_in + 1 / 10 ** (n_in + 1), n_in)
        else:
            return round(a_in, n_in)

    def amount_to_precision(self, symbol, amount_value):
        settings = self.describe()
        amount_precision = settings['precision']['amount']
        amount = self.school_round(amount_value, amount_precision)
        return amount
    def price_to_precision(self, symbol, price_value):
        settings = self.describe()
        price_precision = settings['precision']['price']
        price = self.school_round(price_value, price_precision)
        return price

    async def cancel_order(self, id, symbol=None, params={}):
        await self.load_markets()
        request = {}
        if symbol is not None:
            market = self.market(symbol)
            request = {
                'sym': market['id'],
                'id': id,
                'sd': params['sd'],
            }
        else:
            request = {
                'hash': id,
            }
        return self.privatePostApiMarketV2CancelOrder(self.extend(request, params))

    def fetch_my_trades(self, symbol, since=None, limit=None, params={}):
        self.load_markets()
        market = self.market(symbol)
        request = {
            'sym': market['id'],
        }
        if since is not None:
            request['start'] = int(since / 1000)
            request['end'] = int(self.milliseconds() / 1000)
        if limit is not None:
            request['limit'] = limit
        response = self.privatePostApiMarketMyOrderHistory(self.extend(request, params))
        trades = response['result']
        result = []
        for i in range(0, len(trades)):
            id = self.safe_string(trades[i], 'order_id')
            order = self.safe_string(trades[i], 'txn_id')
            type = self.safe_string(trades[i], 'type')
            side = self.safe_string(trades[i], 'side')
            takerOrMaker = self.safe_value(trades[i], 'taken_by_me')
            price = self.safe_float(trades[i], 'rate')
            amount = self.safe_float(trades[i], 'amount')
            cost = float(price * amount)
            fee = self.safe_float(trades[i], 'fee')
            timestamp = self.safe_timestamp(trades[i], 'ts')
            clientOrderId = self.safe_string(trades[i], 'client_id', '')
            result.append({
                'info': trades[i],
                'id': id,
                'timestamp': timestamp,
                'datetime': self.iso8601(timestamp),
                'symbol': symbol,
                'order': order,
                'type': type,
                'side': side,
                'takerOrMaker': takerOrMaker == 'taker' if True else 'maker',
                'price': price,
                'amount': amount,
                'cost': cost,
                'fee': fee,
                'positionSide': 'spot',
                'clientOrderId': clientOrderId,
            })
        return result

    def parse_trade(self, trade, market=None):
        timestamp = int(trade[0]) * 1000
        side = None
        side = trade[3].lower()
        price = float(trade[1])
        amount = float(trade[2])
        cost = float(price * amount)
        return {
            'info': trade,
            'timestamp': timestamp,
            'datetime': self.iso8601(timestamp),
            'symbol': market['symbol'],
            'id': None,
            'order': None,
            'type': None,
            'takerOrMaker': None,
            'side': side,
            'price': price,
            'amount': amount,
            'cost': cost,
            'fee': None,
        }

    def fetch_trades(self, symbol, since=None, limit=None, params={}):
        self.load_markets()
        market = self.market(symbol)
        request = {
            'sym': market['id'],
        }
        if limit is None:
            limit = 1
        request['lmt'] = limit
        response = self.publicGetApiMarketTrades(self.extend(request, params))
        trades = response['result']
        return self.parse_trades(trades, market, since, limit)

    def fetch_deposit_address(self, code, params={}):
        self.load_markets()
        response = self.privatePostApiCryptoAddresses(params)
        accounts = response['result']
        currency = None
        address = None
        tag = None
        for i in range(0, len(accounts)):
            currency = self.safe_string(accounts[i], 'currency')
            if code == currency:
                address = self.safe_string(accounts[i], 'address')
                tag = self.safe_string(accounts[i], 'tag')
                break
        return {
            'currency': currency,
            'address': address,
            'tag': tag,
            'info': accounts,
        }

    def nonce(self):
        return self.milliseconds()

    def sign(self, path, api='public', method='GET', params={}, headers=None, body=None):
        url = '/' + path
        if api == 'public':
            if params:
                url += '?' + self.urlencode(params)
        elif api == 'private':
            self.check_required_credentials()
            query = self.extend(params, {
                'ts': self.nonce(),
            })
            request = self.json(query)
            signature = self.hmac(self.encode(request), self.encode(self.secret))
            body = self.json(self.extend(json.loads(request), {'sig': signature}))
            headers = {
                'X-BTK-APIKEY': self.apiKey,
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            }
        else:
            url = '/' + path
        url = self.urls['api'] + url
        return {'url': url, 'method': method, 'body': body, 'headers': headers}