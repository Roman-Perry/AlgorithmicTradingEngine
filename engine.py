from __future__ import annotations

import asyncio
import math
import random
import time
import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AlgoEngine")

# --------------------------------------------------
# Events
# --------------------------------------------------

class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass(frozen=True, slots=True)
class MarketEvent:
    symbol: str
    timestamp: float
    price: float
    bid: float
    ask: float
    volume: float
    event_type: EventType = field(default=EventType.MARKET, init=False, compare=False)

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) * 0.5
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def book_imbalance(self) -> float:
        denom = self.spread + 1e-12
        return (self.price - self.mid_price) / denom
    

@dataclass(frozen=True, slots=True)
class SignalEvent:
    symbol: str
    timestamp: float
    signal: int
    strength: float
    gamma_sig: int 
    regime_id: int
    regime_sig: int
    regime_stb: float
    ml_sig: int
    ml_conf: float
    event_type: EventType = field(default=EventType.SIGNAL, init=False, compare=False)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    symbol: str
    timestamp: float
    order_side: OrderSide
    order_type: OrderType
    quantity: float
    order_id: str
    limit_price: Optional[float] = None
    event_type: EventType = field(default=EventType.ORDER, init=False, compare=False)


@dataclass(frozen=True, slots=True)
class FillEvent:
    symbol: str
    timestamp: float
    order_id: str
    order_side: OrderSide
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    event_type: EventType = field(default=EventType.FILL, init=False, compare=False)


AnyEvent = MarketEvent | SignalEvent | OrderEvent | FillEvent

# --------------------------------------------------
# Data Ingestion
# --------------------------------------------------


class MarketDataSimulator:

    def __init__(
            self,
            symbol: str = "SYN",
            initial_price: float = 150.0,
            mu: float = 5e-5,
            sigma_bar: float = 0.0018,
            kappa: float = 0.08,
            xi: float = 0.00008,
            tick_interval: float = 0.05,
            seed: int = 42
        ) -> None:

        self.symbol = symbol
        self.price = initial_price
        self.mu = mu
        self.sigma_bar = sigma_bar
        self.sigma = sigma_bar
        self.kappa = kappa
        self.xi = xi
        self.tick_interval = tick_interval
        self._rng = np.random.default_rng(seed)
        self._vol_history: Deque[float] = deque(maxlen=100)

    
    def _step_vol(self) -> None:
        dt = self.tick_interval
        dW2 = self._rng.standard_normal()
        d_sigma = (
            self.kappa * (self.sigma_bar - self.sigma) * dt +
            self.xi * math.sqrt(dt) * dW2
        )
        self.sigma = max(1e-5, self.sigma + d_sigma)
        self._vol_history.append(self.sigma)

    def _next_tick(self) -> MarketEvent:
        self._step_vol()
        dt = self.tick_interval
        dW1 = self._rng.standard_normal()

        # GBM log price
        log_return = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * math.sqrt(dt) * dW1
        self.price += math.exp(log_return)

        # Bid-ask spread: base + vol premium
        half_spread = self.price * (0.00008 + self.sigma * 0.8)
        bid = self.price - half_spread
        ask = self.price + half_spread

        # Volume: log-normal with vol-regime scaling
        vol_scale = 1.0 + 4.0 * (self.sigma / self.sigma_bar - 1.0) * 0.3
        volume = float(self._rng.lognormal(mean=7.8, sigma=0.7)) * max(0.2, vol_scale)

        return MarketEvent(
            symbol = self.symbol,
            timestamp = time.monotonic(),
            price = round(self.price, 4),
            bid = round(bid, 4),
            ask = round(ask, 4),
            volume = round(volume, 2),
        )
    
    async def stream(self, queue: asyncio.Queue[AnyEvent]) -> None:
        logger.info(f"[DataFeed] Streaming {self.symbol} price={self.price:.4f}")
        while True:
            await queue.put(self._next_tick())
            await asyncio.sleep(self.tick_interval)





        









