import json
from copy import copy
from datetime import datetime
from howtrader.trader.utility import round_to
from typing import Dict, List, Tuple, Any
from howtrader.api.rest import RestClient, Request, Response
from howtrader.trader.constant import (
    Direction,
    Exchange,
    Status,
    OrderType,
    Interval,
    Product,
)
from howtrader.trader.gateway import BaseGateway
from howtrader.trader.object import (
    TickData,
    OrderData,
    TradeData,
    AccountData,
    ContractData,
    BarData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest
)
from howtrader.event import EventEngine
from .huobi_apibase import _split_url, generate_datetime, create_signature, LOCAL_TZ, HuobiWebsocketApiBase

#learning??
# import hashlib
# import hmac
# import time
# from copy import copy
# from datetime import datetime, timedelta
# from enum import Enum
# from threading import Lock
# from typing import Any, Callable, Dict, List, Tuple, Union
# from decimal import Decimal
# import pandas as pd
# import json

# #newfunction
# from howtrader.trader.event import EVENT_TIMER
# from howtrader.event import EventEngine, Event
# from howtrader.api.rest import Request, RestClient, Response
# from howtrader.api.websocket import WebsocketClient
# from howtrader.trader.constant import LOCAL_TZ
# from howtrader.trader.setting import SETTINGS
# import base64
# import gzip
# import logging
# import sys
# import traceback
# from collections import defaultdict
# from urllib.parse import urlencode

# REST API HOST
REST_HOST = "https://api.huobi.pro"

# Websocket API HOST
WEBSOCKET_DATA_HOST = "wss://api.huobi.pro/ws"
WEBSOCKET_TRADE_HOST = "wss://api.huobi.pro/ws/v2"

# # order status mapping
# STATUS_HUOBI2VT: Dict[str, Status] = {
#     "submitted": Status.SUBMITTING,
#     "partial-filled": Status.PARTTRADED,
#     "filled": Status.ALLTRADED,
#     "canceled": Status.CANCELLED,
#     "rejected": Status.REJECTED,
# }
# 委托状态映射
STATUS_HUOBI2VT: Dict[str, Status] = {
    "submitted": Status.NOTTRADED,
    "partial-filled": Status.PARTTRADED,
    "filled": Status.ALLTRADED,
    "cancelling": Status.CANCELLED,
    "partial-canceled": Status.CANCELLED,
    "canceled": Status.CANCELLED,
}
# # order type mapping
# ORDERTYPE_VT2HUOBI: Dict[Tuple[Direction, OrderType], str] = {
#     OrderType.LIMIT: "LIMIT",
#     OrderType.TAKER: "MARKET",
#     OrderType.MAKER: "LIMIT_MAKER",
# }
#
# ORDERTYPE_HUOBI2VT: Dict[str, OrderType] = {v: k for k, v in ORDERTYPE_VT2HUOBI.items()}
# order direction mapping
# DIRECTION_VT2HUOBI: Dict[Direction, str] = {
#     Direction.LONG: "BUY",
#     Direction.SHORT: "SELL"
# }
# DIRECTION_HUOBI2VT: Dict[str, Direction] = {v: k for k, v in DIRECTION_VT2HUOBI.items()}

# 委托类型映射
ORDERTYPE_VT2HUOBI: Dict[Tuple[Direction, OrderType], str] = {
    (Direction.LONG, OrderType.TAKER): "buy-market",
    (Direction.SHORT, OrderType.TAKER): "sell-market",
    (Direction.LONG, OrderType.LIMIT): "buy-limit",
    (Direction.SHORT, OrderType.LIMIT): "sell-limit",
    (Direction.LONG, OrderType.MAKER): "buy-limit-maker",
    (Direction.SHORT, OrderType.MAKER): "sell-limit-maker",
}
ORDERTYPE_HUOBI2VT: Dict[str, Tuple[Direction, OrderType]] = {v: k for k, v in ORDERTYPE_VT2HUOBI.items()}

# # data timeframe mapping
# INTERVAL_VT2HUOBI: Dict[Interval, str] = {
#     Interval.MINUTE: "1m",
#     Interval.MINUTE_3: "3m",
#     Interval.MINUTE_5: "5m",
#     Interval.MINUTE_15: "15",
#     Interval.MINUTE_30: "30m",
#     Interval.HOUR: "1h",
#     Interval.HOUR_2: "2h",
#     Interval.HOUR_4: "4h",
#     Interval.HOUR_6: "6h",
#     Interval.HOUR_8: "8h",
#     Interval.HOUR_12: "12h",
#     Interval.DAILY: "1d",
#     Interval.DAILY_3: "3d",
#     Interval.WEEKLY: "1w",
#     Interval.MONTH: "1M"
# }
# 数据频率映射
INTERVAL_VT2HUOBI: Dict[Interval, str] = {
    Interval.MINUTE: "1min",
    Interval.HOUR: "60min",
    Interval.DAILY: "1day"
}

# time delta mapping
# TIMEDELTA_MAP: Dict[Interval, timedelta] = {
#     Interval.MINUTE: timedelta(minutes=1),
#     Interval.HOUR: timedelta(hours=1),
#     Interval.DAILY: timedelta(days=1),
# }

# contract mapping
symbol_contract_map: Dict[str, ContractData] = {}
# 币种余额全局缓存字典
currency_balance: Dict[str, float] = {}

class HuobiSpotGateway(BaseGateway):
    """
    huobi spot gateway for howtrader
    """

    default_name: str = "HUOBI_SPOT"

    default_setting: Dict[str, Any] = {
        "key": "",
        "secret": "",
        "proxy_host": "",
        "proxy_port": 0,
    }

    exchanges = [Exchange.HUOBI]

    def __init__(self, event_engine: EventEngine, gateway_name: str)-> None:
        """init"""
        super().__init__(event_engine, gateway_name)

        self.trade_ws_api: "HuobiSpotTradeWebsocketApi" = HuobiSpotTradeWebsocketApi(self)
        self.market_ws_api:"HuobiSpotDataWebsocketApi" = HuobiSpotDataWebsocketApi(self)
        self.rest_api: "HuobiSpotRestApi" = HuobiSpotRestApi(self)

        self.orders: Dict[str, OrderData] = {}
        self.get_server_time_interval: int = 0

    def connect(self, setting: dict):
        """connect huobi api"""
        key: str = setting["key"]
        secret: str = setting["secret"]

        if isinstance(setting["proxy_host"], str):
            proxy_host: str = setting["proxy_host"]
        else:
            proxy_host: str = ""

        if isinstance(setting["proxy_port"], int):
            proxy_port: int = setting["proxy_port"]
        else:
            proxy_port: int = 0

        self.rest_api.connect(key, secret, proxy_host, proxy_port)
        self.market_ws_api.connect(key, secret, proxy_host, proxy_port)
        self.trade_ws_api.connect(key, secret, proxy_host, proxy_port)


    def subscribe(self, req: SubscribeRequest) -> None:
        """subscribe market data"""
        self.market_ws_api.subscribe(req)

    def send_order(self, req: OrderRequest) -> str:
        """place order"""
        return self.rest_api.send_order(req)

    def cancel_order(self, req: CancelRequest) -> None:
        """cancel order"""
        self.rest_api.cancel_order(req)

    def query_account(self) -> None:
        """query account data"""
        # self.rest_api.query_account()
        pass

    def query_position(self) -> None:
        """query position, not available for spot gateway"""
        pass

    # def query_order(self, req: OrderQueryRequest) -> None:
    #     self.rest_api.query_order(req)

    def query_history(self, req: HistoryRequest) -> List[BarData]:
        """query historical kline data"""
        return self.rest_api.query_history(req)

    def close(self) -> None:
        """close connection from exchange server"""
        self.rest_api.stop()
        self.trade_ws_api.stop()
        self.market_ws_api.stop()

    # def process_timer_event(self, event: Event) -> None:
    #     """process timer event, for updating the listen key"""
    #     self.rest_api.keep_user_stream()
    #     self.get_server_time_interval += 1
    #
    #     if self.get_server_time_interval >= SETTINGS.get('update_server_time_interval', 300):
    #         self.rest_api.query_time()
    #         self.get_server_time_interval = 0

    # def on_order(self, order: OrderData) -> None:
    #     """on order, order update"""
    #     order.update_time = generate_datetime(time.time() * 1000)
    #     last_order: OrderData = self.get_order(order.orderid)
    #     if not last_order:
    #         self.orders[order.orderid] = order
    #         super().on_order(copy(order))
    #
    #     else:
    #         traded: Decimal = order.traded - last_order.traded
    #         if traded < 0: # filter the order is not in sequence
    #             return None
    #
    #         if traded > 0:
    #             trade: TradeData = TradeData(
    #                 symbol=order.symbol,
    #                 exchange=order.exchange,
    #                 orderid=order.orderid,
    #                 direction=order.direction,
    #                 price=order.traded_price,
    #                 volume=traded,
    #                 datetime=order.update_time,
    #                 gateway_name=self.gateway_name,
    #             )
    #             super().on_trade(trade)
    #
    #         if traded == 0 and order.status == last_order.status:
    #             return None
    #
    #         self.orders[order.orderid] = order
    #         super().on_order(copy(order))
    #
    # def get_order(self, orderid: str) -> OrderData:
    #     """get order by orderid"""
    #     return self.orders.get(orderid, None)
    def on_order(self, order: OrderData) -> None:
        """推送委托数据,same"""
        self.orders[order.orderid] = order
        super().on_order(order)

    def get_order(self, orderid: str) -> OrderData:
        """查询委托数据"""
        return self.orders.get(orderid, None)

class HuobiSpotRestApi(RestClient):
    """huobi spot rest api"""
    def __init__(self, gateway: HuobiSpotGateway)-> None:
        """init"""
        super().__init__()

        self.gateway: HuobiSpotGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.host: str = ""
        self.key: str = ""
        self.secret: str = ""
        self.account_id: str = ""

        self.order_count: int = 0

        # self.user_stream_key: str = ""
        # self.keep_alive_count: int = 0
        # self.keep_alive_failed_count: int = 0
        # self.recv_window: int = 5000
        # self.time_offset: int = 0
        #
        # self.order_count: int = 1_000_000
        # self.order_count_lock: Lock = Lock()
        # self.connect_time: int = 0

    def sign(self, request: Request) -> Request:
        """signature for private api"""
        request.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36"
        }
        params_with_signature = create_signature(
            self.key,
            request.method,
            self.host,
            request.path,
            self.secret,
            request.params
        )
        request.params = params_with_signature

        if request.method == "POST":
            request.headers["Content-Type"] = "application/json"

            if request.data:
                request.data = json.dumps(request.data)

        return request

    def connect(
            self,
            key: str,
            secret: str,
            proxy_host: str,
            proxy_port: int
    ) -> None:
        """连接REST服务器"""
        self.key = key
        self.secret = secret
        self.host, _ = _split_url(REST_HOST)

        self.init(REST_HOST, proxy_host, proxy_port)
        self.start()

        self.gateway.write_log("start connecting rest api")

        self.query_contract()
        self.query_account()
        self.query_order()

    def query_account(self) -> None:
        """查询资金"""
        self.add_request(
            method="GET",
            path="/v1/account/accounts",
            callback=self.on_query_account
        )

    def query_order(self) -> None:
        """查询未成交委托"""
        self.add_request(
            method="GET",
            path="/v1/order/openOrders",
            callback=self.on_query_order
        )

    def query_contract(self) -> None:
        """查询合约信息"""
        self.add_request(
            method="GET",
            path="/v1/common/symbols",
            callback=self.on_query_contract
        )

    def new_orderid(self) -> str:
        """生成本地委托号"""
        prefix: str = datetime.now().strftime("%Y%m%d-%H%M%S-")

        self.order_count += 1
        suffix: str = str(self.order_count).rjust(8, "0")

        orderid: str = prefix + suffix
        return orderid

    def send_order(self, req: OrderRequest) -> str:
        """委托下单"""
        huobi_type: str = ORDERTYPE_VT2HUOBI.get(
            (req.direction, req.type), ""
        )

        orderid: str = self.new_orderid()
        order: OrderData = req.create_order_data(
            orderid,
            self.gateway_name)
        order.datetime = datetime.now(LOCAL_TZ)

        data: dict = {
            "account-id": self.account_id,
            "amount": str(req.volume),
            "symbol": req.symbol,
            "type": huobi_type,
            "price": str(req.price),
            "source": "api",
            "client-order-id": orderid
        }

        self.add_request(
            method="POST",
            path="/v1/order/orders/place",
            callback=self.on_send_order,
            data=data,
            extra=order,
            on_error=self.on_send_order_error,
            on_failed=self.on_send_order_failed
        )

        self.gateway.on_order(order)
        return order.vt_orderid

    def cancel_order(self, req: CancelRequest) -> None:
        """委托撤单"""
        data: dict = {"client-order-id": req.orderid}

        self.add_request(
            method="POST",
            path="/v1/order/orders/submitCancelClientOrder",
            callback=self.on_cancel_order,
            on_failed=self.on_cancel_order_failed,
            data=data,
            extra=req
        )

    def on_query_account(self, data: dict, request: Request) -> None:
        """资金查询回报"""
        if self.check_error(data, "查询账户"):
            return

        for d in data["data"]:
            if d["type"] == "spot":
                self.account_id = d["id"]
                self.gateway.write_log(f"账户代码{self.account_id}查询成功")

    def on_query_order(self, data: dict, request: Request) -> None:
        """未成交委托查询回报"""
        if self.check_error(data, "查询委托"):
            return

        for d in data["data"]:
            direction, order_type = ORDERTYPE_HUOBI2VT[d["type"]]

            order: OrderData = OrderData(
                orderid=d["client-order-id"],
                symbol=d["symbol"],
                exchange=Exchange.HUOBI,
                price=float(d["price"]),
                volume=float(d["amount"]),
                type=order_type,
                direction=direction,
                traded=float(d["filled-amount"]),
                status=STATUS_HUOBI2VT.get(d["state"], None),
                datetime=generate_datetime(d["created-at"] / 1000),
                gateway_name=self.gateway_name,
            )

            self.gateway.on_order(order)

        self.gateway.write_log("委托信息查询成功")

    def on_query_contract(self, data: dict, request: Request) -> None:
        """合约信息查询回报"""
        if self.check_error(data, "查询合约"):
            return

        for d in data["data"]:
            base_currency: str = d["base-currency"]
            quote_currency: str = d["quote-currency"]
            name: str = f"{base_currency.upper()}/{quote_currency.upper()}"

            pricetick: float = 1 / pow(10, d["price-precision"])
            min_volume: float = 1 / pow(10, d["amount-precision"])

            contract: ContractData = ContractData(
                symbol=d["symbol"],
                exchange=Exchange.HUOBI,
                name=name,
                pricetick=pricetick,
                size=1,
                min_volume=min_volume,
                product=Product.SPOT,
                history_data=True,
                gateway_name=self.gateway_name,
            )
            self.gateway.on_contract(contract)

            symbol_contract_map[contract.symbol] = contract

        self.gateway.write_log("合约信息查询成功")

    def on_send_order(self, data: dict, request: Request) -> None:
        """委托下单回报"""
        order: OrderData = request.extra

        if self.check_error(data, "委托"):
            order.status = Status.REJECTED
            self.gateway.on_order(order)

    def on_send_order_failed(self, status_code: str, request: Request) -> None:
        """委托下单失败服务器报错回报"""
        order: OrderData = request.extra
        order.status = Status.REJECTED
        self.gateway.on_order(order)

        msg: str = f"委托失败，状态码：{status_code}，信息：{request.response.text}"
        self.gateway.write_log(msg)

    def on_send_order_error(
            self,
            exception_type: type,
            exception_value: Exception,
            tb,
            request: Request
    ) -> None:
        """委托下单回报函数报错回报"""
        order: OrderData = request.extra
        order.status = Status.REJECTED
        self.gateway.on_order(order)

        if not issubclass(exception_type, ConnectionError):
            self.on_error(exception_type, exception_value, tb, request)

    def on_cancel_order(self, data: dict, request: Request) -> None:
        """委托撤单回报"""
        self.check_error(data, "撤单")

    def on_cancel_order_failed(self, status_code: str, request: Request) -> None:
        """委托撤单失败服务器报错回报"""
        msg: str = f"撤单失败，状态码：{status_code}，信息：{request.response.text}"
        self.gateway.write_log(msg)

    def on_error(
            self,
            exception_type: type,
            exception_value: Exception,
            tb,
            request: Request
    ) -> None:
        """请求触发异常的回调"""
        msg: str = f"触发异常，状态码：{exception_type}，信息：{exception_value}"
        self.gateway.write_log(msg)

        super().on_error(exception_type, exception_value, tb, request)

    def check_error(self, data: dict, func: str = "") -> bool:
        """回报状态检查"""
        if data["status"] != "error":
            return False

        error_code: str = data["err-code"]
        error_msg: str = data["err-msg"]

        self.gateway.write_log(f"{func}请求出错，代码：{error_code}，信息：{error_msg}")
        return True

    def query_history(self, req: HistoryRequest) -> List[BarData]:
        """查询历史数据"""
        # 创建查询参数
        params: dict = {
            "symbol": req.symbol,
            "period": INTERVAL_VT2HUOBI[req.interval],
            "size": 2000
        }

        resp: Response = self.request(
            "GET",
            "/market/history/kline",
            params=params
        )

        # 如果请求失败则终止循环
        history: List[BarData] = []

        if resp.status_code // 100 != 2:
            msg: str = f"获取历史数据失败，状态码：{resp.status_code}，信息：{resp.text}"
            self.gateway.write_log(msg)
        else:
            data: dict = resp.json()
            if not data:
                msg: str = f"获取历史数据为空"
                self.gateway.write_log(msg)
            else:
                for d in data["data"]:
                    bar = BarData(
                        symbol=req.symbol,
                        exchange=req.exchange,
                        datetime=generate_datetime(d["id"]),
                        interval=req.interval,
                        volume=d["vol"],
                        open_price=d["open"],
                        high_price=d["high"],
                        low_price=d["low"],
                        close_price=d["close"],
                        gateway_name=self.gateway_name
                    )
                    history.append(bar)

                history.reverse()
                begin: datetime = history[0].datetime
                end: datetime = history[-1].datetime
                msg: str = f"获取历史数据成功，{req.symbol} - {req.interval.value}，{begin} - {end}"
                self.gateway.write_log(msg)

        return history

class HuobiSpotTradeWebsocketApi(HuobiWebsocketApiBase):
    """火币现货交易Websocket API"""

    def __init__(self, gateway: HuobiSpotGateway):
        """构造函数"""
        super().__init__(gateway)

    def connect(
        self,
        key: str,
        secret: str,
        proxy_host: str,
        proxy_port: int
    ) -> None:
        """连接Websocket交易频道"""
        super().connect(
            key,
            secret,
            WEBSOCKET_TRADE_HOST,
            proxy_host,
            proxy_port
        )

    def subscribe_topic(self) -> None:
        """订阅委托和资金推送"""
        req: dict = {
            "action": "sub",
            "ch": f"orders#*"
        }
        self.send_packet(req)

        req: dict = {
            "action": "sub",
            "ch": "accounts.update#1"
        }
        self.send_packet(req)

    def on_connected(self) -> None:
        """连接成功回报"""
        self.gateway.write_log("交易Websocket API连接成功")
        self.login(v2=True)

    def on_login(self, packet: dict) -> None:
        """登录成功回报"""
        if "data" in packet and not packet["data"]:
            self.gateway.write_log("交易Websocket API登录成功")

            self.subscribe_topic()
        else:
            msg: str = packet["message"]
            error_msg: str = f"交易Websocket API登录失败，原因：{msg}"
            self.gateway.write_log(error_msg)

    def on_data(self, packet: dict) -> None:
        """推送数据回报"""
        action: str = packet.get("action", None)
        if action and action != "sub":

            ch: str = packet["ch"]
            if "orders" in ch:
                self.on_order(packet["data"])
            elif "accounts" in ch:
                self.on_account(packet["data"])

    def on_account(self, data: dict) -> None:
        """资金更新推送"""
        if not data:
            return

        currency: str = data["currency"]

        change_type: str = data["changeType"]
        if not change_type:
            balance: float = float(data["balance"])
            frozen: float = balance - float(data["available"])
            currency_balance[currency] = balance

        elif "place" in change_type:
            if "available" not in data:
                return
            balance: float = currency_balance[currency]
            frozen: float = balance - float(data["available"])
        else:
            frozen: float = 0.0
            if "balance" in data:
                balance: float = float(data["balance"])
            else:
                balance: float = float(data["available"])
            currency_balance[currency] = balance

        account: AccountData = AccountData(
            accountid=currency,
            balance=balance,
            frozen=frozen,
            gateway_name=self.gateway_name,
        )
        self.gateway.on_account(account)

    def on_order(self, data: dict) -> None:
        """委托更新推送"""
        orderid: str = data["clientOrderId"]
        order: OrderData = self.gateway.get_order(orderid)
        if not order:
            return

        # 将成交数量四舍五入到正确精度
        traded_volume: float = float(data.get("tradeVolume", 0))
        contract: ContractData = symbol_contract_map.get(order.symbol, None)
        if contract:
            traded_volume = round_to(traded_volume, contract.min_volume)

        order.traded += traded_volume
        order.status = STATUS_HUOBI2VT.get(data["orderStatus"], None)
        self.gateway.on_order(order)

        if not traded_volume:
            return

        trade: TradeData = TradeData(
            symbol=order.symbol,
            exchange=Exchange.HUOBI,
            orderid=order.orderid,
            tradeid=str(data["tradeId"]),
            direction=order.direction,
            price=float(data["tradePrice"]),
            volume=traded_volume,
            datetime=datetime.now(LOCAL_TZ),
            gateway_name=self.gateway_name,
        )
        self.gateway.on_trade(trade)

class HuobiSpotDataWebsocketApi(HuobiWebsocketApiBase):
    """火币现货行情Websocket API"""

    def __init__(self, gateway: HuobiSpotGateway):
        """构造函数"""
        super().__init__(gateway)

        self.ticks: Dict[str, TickData] = {}
        self.subscribed: Dict[str, SubscribeRequest] = {}

    def connect(
        self,
        key: str,
        secret: str,
        proxy_host: str,
        proxy_port: int
    ) -> None:
        """连接Websocket行情频道"""
        super().connect(
            key,
            secret,
            WEBSOCKET_DATA_HOST,
            proxy_host,
            proxy_port
        )

    def on_connected(self) -> None:
        """连接成功回报"""
        self.gateway.write_log("行情Websocket API连接成功")

        for req in list(self.subscribed.values()):
            self.subscribe(req)

    def subscribe(self, req: SubscribeRequest) -> None:
        """订阅行情"""
        if req.symbol not in symbol_contract_map:
            self.gateway.write_log(f"找不到该合约代码{req.symbol}")
            return

        # 缓存订阅记录
        self.subscribed[req.vt_symbol] = req

        # 创建TICK对象
        tick: TickData = TickData(
            symbol=req.symbol,
            name=symbol_contract_map[req.symbol].name,
            exchange=Exchange.HUOBI,
            datetime=datetime.now(LOCAL_TZ),
            gateway_name=self.gateway_name,
        )
        self.ticks[req.symbol] = tick

        # 发送订阅请求
        req_dict: dict = {
            "sub": f"market.{req.symbol}.depth.step0"
        }
        self.send_packet(req_dict)

        req_dict: dict = {
            "sub": f"market.{req.symbol}.detail"
        }
        self.send_packet(req_dict)

    def on_data(self, packet: dict) -> None:
        """推送数据回报"""
        channel: str = packet.get("ch", None)
        if channel:
            if "depth.step" in channel:
                self.on_market_depth(packet)
            elif "detail" in channel:
                self.on_market_detail(packet)
        elif "err-code" in packet:
            code: str = packet["err-code"]
            msg: str = packet["err-msg"]
            self.gateway.write_log(f"错误代码：{code}, 错误信息：{msg}")

    def on_market_depth(self, data: dict) -> None:
        """行情深度推送 """
        symbol: str = data["ch"].split(".")[1]
        tick: TickData = self.ticks[symbol]
        tick.datetime = generate_datetime(data["ts"] / 1000)

        bids: list = data["tick"]["bids"]
        for n in range(min(5, len(bids))):
            price, volume = bids[n]
            tick.__setattr__("bid_price_" + str(n + 1), float(price))
            tick.__setattr__("bid_volume_" + str(n + 1), float(volume))

        asks: list = data["tick"]["asks"]
        for n in range(min(5, len(asks))):
            price, volume = asks[n]
            tick.__setattr__("ask_price_" + str(n + 1), float(price))
            tick.__setattr__("ask_volume_" + str(n + 1), float(volume))

        if tick.last_price:
            tick.localtime = datetime.now()
            self.gateway.on_tick(copy(tick))

    def on_market_detail(self, data: dict) -> None:
        """市场细节推送"""
        symbol: str = data["ch"].split(".")[1]
        tick: TickData = self.ticks[symbol]
        tick.datetime = generate_datetime(data["ts"] / 1000)

        tick_data = data["tick"]
        tick.open_price = float(tick_data["open"])
        tick.high_price = float(tick_data["high"])
        tick.low_price = float(tick_data["low"])
        tick.last_price = float(tick_data["close"])
        tick.volume = float(tick_data["amount"])
        tick.turnover = float(tick_data["vol"])

        if tick.bid_price_1:
            tick.localtime = datetime.now()
            self.gateway.on_tick(copy(tick))