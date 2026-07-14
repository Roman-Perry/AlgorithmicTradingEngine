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

# --------------------------------------------------
# Strategy
# --------------------------------------------------


class GammaSignalEngine:

    def __init__(
            self,
            window: int = 80,
            upper_pct: float = 0.82,
            lower_pct: float = 0.18,
    ) -> None:
        self._window = window
        self._upper_pct = upper_pct
        self._lower_pct = lower_pct
        self._iats: Deque[float] = deque(maxlen=window)
        self._last_ts: Optional[float] = None
        self._alpha: float = 2.0
        self._beta: float = 1.0
        self._signal: int = 0
        self._strength: float = 0.0

    # Parameter Estimate
    def _fit_gamma(self, data: np.ndarray) -> Tuple[float, float]:
        if len(data) < 12 or data.std() < 1e-12:
            return 2.0, 1.0
        try:
            alpha_hat, _loc, scale_hat = stats.gamma.fit(data, floc=0.0)
        except Exception:
            return 2.0, 1.0
        
    # Public interface
    def update(self, event: MarketEvent) -> int:
        now = event.timestamp
        if self._last_ts is not None:
            iat = max(1e-9, now - self._last_ts)
            self._iats.append(iat)
        self._last_ts = now

        min_obs = self._window // 3
        if len(self._iats) < min_obs:
            self._signal = 0
            self._strength = 0.0
            return 0
        
        arr = np.asarray(self._iats, dtype=np.float64)
        self._alpha, self._beta = self._fit_gamma(arr)

        # CDF evaluation
        current_iat = self._iats[-1]
        scale = 1.0 / self._beta
        cdf_val = float(stats.gamma.cdf(current_iat, a=self._alpha, scale=scale)) 

        # PDF ratio
        mode = max(0.0, (self._alpha - 1.0) * scale)
        pdf_curr = stats.gamma.pdf(current_iat, a=self._alpha, scale=scale)
        pdf_mode = stats.gamma.pdf(mode, a=self._alpha, scale=scale) if mode > 0 else 1e-10
        pdf_ratio = pdf_curr / (pdf_mode + 1e-12)

        # Tail conditions
        in_upper_tail = cdf_val > self._upper_pct and pdf_ratio < 0.40
        in_lower_tail = cdf_val < self._lower_pct

        if in_upper_tail:
            self._signal = 1
        elif in_lower_tail:
            self._signal = -1
        else:
            self._signal = 0

        boundary = self._upper_pct - 0.5
        self._strength = min(1.0, abs(cdf_val - 0.5) / (boundary + 1e-9))
        return self._signal
    
    @property
    def strength(self) -> float:
        return self._strength
    
    @property
    def fitted_alpha(self) -> float:
        return self._alpha


class RegimeClusterEngine:

    REGIME_LABELS: Dict[int, str] = {
        0: "LowVol-Bull",
        1: "HighVol-Bear",
        2: "Sideways",
    }

    def __init__(
            self,
            window: int = 160,
            n_clusters: int = 3, 
            refit_every: int = 30,
    ) -> None:
        self._window = window
        self._n_clusters = n_clusters
        self._refit_every = refit_every
        self._kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, n_init=5, batch_size=64
        )
        self._scalar = StandardScaler()
        self._prices: Deque[float] = deque(maxlen=window)
        self._volumes: Deque[float] = deque(maxlen=window)
        self._regime_hist: Deque[int] = deque(maxlen=30)
        self._tick_count: int = 0
        self._fitted: bool = False
        self._regime_map: Dict[int, int] = {0: 2, 1: 2, 2: 2}
        self._current: int = 2
        self._stability: float = 0.5

        # Regime features


        
        


        









