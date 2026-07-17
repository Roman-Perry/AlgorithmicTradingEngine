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
from sklearn.ensemble import GradientBoostingClassifier
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
        self._scaler = StandardScaler()
        self._prices: Deque[float] = deque(maxlen=window)
        self._volumes: Deque[float] = deque(maxlen=window)
        self._regime_hist: Deque[int] = deque(maxlen=30)
        self._tick_count: int = 0
        self._fitted: bool = False
        self._regime_map: Dict[int, int] = {0: 2, 1: 2, 2: 2}
        self._current: int = 2
        self._stability: float = 0.5

        
    # Regime features
    @staticmethod
    def _rsi(prices: np.ndarray, period: int = 14) -> float:
        n = len(prices)
        if n < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        up = deltas.clip(min=0)
        down = (-deltas).clip(min=0)
        avg_up = up.mean() + 1e-12
        avg_dn = down.mean() + 1e-12
        return 100.0 - 100.0 / (1 + avg_up / avg_dn)
    
    def _feature_row(
            self,
            prices: np.ndarray,
            volumes: np.ndarray,
            end_idx: int, 
        ) -> Optional[np.ndarray]:
        if end_idx < 22:
            return None
        p = prices[max(0, end_idx - 22): end_idx + 1]
        v = volumes[max(0, end_idx - 20): end_idx + 1]

        # Rolling 20-tick realized vol
        rets = np.diff(np.log(p[-21:]))
        r_vol = float(np.std(rets)) * math.sqrt(252 * 6.5 * 3600 / 0.05) + 1e-10

        # RSI-14
        rsi = self._rsi(p[-16:] if len(p) >= 16 else p)

        # Volume ratio 
        v_ratio = float(v[-1]) / (float(np.mean(v[-20:])) + 1e-9)

        # 5-tick price momentum
        momentum = (float(p[-1]) / float(p[-6]) - 1.0) * 100.0 if len(p) >= 6 else 0.0

        # 10 tick normalized range
        p10 = p[-10:] if len(p) >= 10 else p
        p_rng = (float(np.max(p10)) - float(np.min(p10))) / (float(np.mean(p10)) + 1e-9) * 100.0

        return np.array([r_vol, rsi, v_ratio, momentum, p_rng], dtype=np.float64)
    

    def _resolve_labels(self) -> None:
        centers_scaled = self._kmeans.cluster_centers_
        centers = self._scaler.inverse_transform(centers_scaled)
        vol_order = np.argsort(centers[:, 0])

        self._regime_map = {
            int(vol_order[0]): 0,
            int(vol_order[2]): 1,
            int(vol_order[1]): 2, 
        }

    # Public interface
    def update(self, event: MarketEvent) -> int:
        self._prices.append(event.price)
        self._volumes.append(event.volume)
        self._tick_count += 1

        prices = np.asarray(self._prices, dtype=np.float64)
        volumes = np.asarray(self._volumes, dtype=np.float64)

        if self._tick_count % self._refit_every == 0 or not self._fitted:
            X_rows: List[np.ndarray] = []
            for i in range(22, len(prices)):
                row = self._feature_row(prices, volumes, i)
                if row is not None:
                    X_rows.append(row)
            if len(X_rows) >= self._n_clusters * 4:
                X = np.vstack(X_rows)
                X_scaled = self._scaler.fit_transform(X)
                self._kmeans.partial_fit(X_scaled)
                self._fitted = True
                self._resolve_labels()

        if not self._fitted:
            return self._current
        
        idx = len(prices) - 1
        row = self._feature_row(prices, volumes, idx)
        if row is None:
            return self._current
        
        row_scaled = self._scaler.transform(row.reshape(1, -1))
        raw_label = int(self._kmeans.predict(row_scaled)[0])
        regime = self._regime_map.get(raw_label, 2)

        self._current = regime
        self._regime_hist.append(regime)

        if len(self._regime_hist) >= 5:
            self._stability = sum(r == regime for r in self._regime_hist) / len(self._regime_hist)

        return self._current
    
    @property
    def regime_signal(self) -> int:
        return {0: 1, 1: -1, 2: 0}[self._current]   

    @property
    def stability(self) -> float:
        return self._stability

    @property
    def current_regime(self) -> int:
        return self._current
    

class SupervisedPredictionEngine:
    
    def __init__(
            self,
            window: int = 200,
            lookahead: int = 4,
            refit_every: int = 40,
            min_fit_samples: int = 90,
            n_estimators: int = 60,
    ) -> None:
        
        self._window = window
        self._lookahead = lookahead
        self.refit_every = refit_every
        self._min_fit_samples = min_fit_samples
        self._n_estimators = n_estimators
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.80,
            max_features="sqrt",
            random_state=7,
        )
        self._scaler = StandardScaler()
        self._prices: Deque[float] = deque(maxlen=window)
        self._bids: Deque[float] = deque(maxlen=window)
        self._asks: Deque[float] = deque(maxlen=window)
        self._volumes: Deque[float] = deque(maxlen=window)
        self._regimes: Deque[int] = deque(maxlen=window)
        self._tick_n: int = 0
        self._fitted: bool = False
        self._signal: int = 0
        self._conf: float = 0.34

    # Feature engineering
    @staticmethod
    def _build_fvec(
        prices: np.ndarray,
        bids: np.ndarray,
        asks: np.ndarray,
        volumes: np.ndarray,
        regimes: np.ndarray,
        idx: int,
    ) -> Optional[np.ndarray]:
        if idx < 21:
            return None
        
        p = prices[:idx + 1]
        b = bids[:idx + 1]
        a = asks[:idx + 1]
        v = volumes[:idx + 1]

        # Lagged returns
        lag_rets: List[float] = []
        for lag in (1, 2, 3, 4, 5, 10):
            if len(p) > lag:
                lag_rets.append((p[-1] / p[-1 - lag] - 1.0) * 1_000.0)
            else: 
                lag_rets.append(0.0)

        # Order book imbalance
        spread = a[-1] - b[-1]
        mid = (a[-1] + b[-1]) * 0.5
        imbalance = (p[-1] - mid) / (spread + 1e-12)

        # Volume ratio
        v_mean = np.mean(v[-20:]) if len(v) >= 20 else v[-1]
        v_ratio = float(v[-1]) / (float(v_mean) + 1e-9)

        # Rolling realized vol
        if len(p) >= 11:
            r10 = np.diff(np.log(p[-11:]))
            r_vol = float(np.std(r10)) * 1_000.0
        else:
            r_vol = 0.0

        # Regime id
        regime = float(regimes[idx]) if idx < len(regimes) else 1.0

        # Price acceleration
        if len(p) >= 3:
            r1 = float(p[-1]) - float(p[-2])
            r2 = float(p[-2]) - float(p[-3])
            accel = (r1 - r2) * 1_000.0
        else:
            accel = 0.0

        return np.array(lag_rets + [imbalance, v_ratio, r_vol, regime, accel], dtype=np.float64)
    
    # Dataset construction
    def _build_dataset(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        n = len(self._prices)
        if n < self._min_fit_samples + self._lookahead:
            return None, None
        
        prices = np.asarray(self._prices, dtype=np.float64)
        bids = np.asarray(self._bids, dtype=np.float64)
        asks = np.asarray(self._asks, dtype=np.float64)
        volumes = np.asarray(self._volumes, dtype=np.float64)
        regimes = np.asarray(self._regimes, dtype=np.float64)

        X_rows: List[np.ndarray] = []
        y_vals: List[int] = []

        for i in range(21, n - self._lookahead):
            fv = self._build_fvec(prices, bids, asks, volumes, regimes, i)
            if fv is None:
                continue
            fwd_ret = prices[i + self._lookahead] / prices[i] - 1.0
            if fwd_ret > 8e-5:
                y_vals.append(1)
            elif fwd_ret < -8e-5:
                y_vals.append(-1)
            else:
                y_vals.append(0)
            X_rows.append(fv)

        if len(X_rows) < self._min_fit_samples:
            return None, None

        return np.vstack(X_rows), np.asarray(y_vals, dtype=np.int64)
    
    # Public interface
    def update(self, event: MarketEvent, regime: int) -> int:
        self._prices.append(event.price)
        self._bids.append(event.bid)
        self._asks.append(event.ask)
        self._volumes.append(event.volume)
        self._regimes.append(regime)
        self._tick_n += 1

        # refit
        if self._tick_n % self._refit_every == 0:
            X, y = self._build_dataset()
            if X is not None and y is not None:
                classes = np.unique(y)
                if len(classes) >= 2:
                    X_sc = self._scaler.fit_transform(X)
                    self._model.fit(X_sc, y)
                    self._fitted = True
        if not self._fitted:
            return 0
        
        # Prediction
        prices = np.asarray(self._prices, dtype=np.float64)
        bids = np.asarray(self._bids, dtype=np.float64)
        asks = np.asarray(self._asks, dtype=np.float64)
        volumes = np.asarray(self._volumes, dtype=np.float64)
        regimes = np.asarray(self._regimes, dtype=np.float64)

        idx = len(prices) - 1
        fv = self._build_fvec(prices, bids, asks, volumes, regimes, idx)
        if fv is None:
            return 0
        fv_sc = self._scaler.transform(fv.reshape(1, -1))
        pred = int(self._model.predict(fv_sc)[0])
        proba = self._model.predict_proba(fv_sc)[0]
        self._conf = float(proba.max())
        self._signal = pred
        
        return pred
    
    @property
    def confidence(self) -> float:
        return self._conf
    

class MetaEnsemble:

    BASE_W_GAMMA: float = 0.25
    BASE_W_REGIME: float = 0.35
    BASE_W_SUPERVISED: float = 0.40
    THRESHOLD: float = 0.28

    def combine(
            self,
            gamma_sig: float,
            gamma_strength: float,
            gamma_alpha: float,
            regime_sig: int,
            regime_stab: float,
            ml_sig: int,
            ml_conf: float,
    ) -> Tuple[int, float]:
        w_g = self.BASE_W_GAMMA
        w_r = self.BASE_W_REGIME
        w_ml = self.BASE_W_SUPERVISED

        # Stability modulation
        stab_delta = (regime_stab - 0.50) * 0.40
        w_r += stab_delta * 0.6
        w_ml += stab_delta * 0.40
        w_g += stab_delta * 0.50

        # ML conf
        if ml_conf > 0.55:
            conf_boost = (ml_conf - 0.55) * 0.50
            w_ml += conf_boost
            w_g -= conf_boost * 0.40
            w_r -= conf_boost * 0.60

        # Gamma sharpness
        if gamma_alpha > 4.0:
            alpha_boost = min(0.10, (gamma_alpha - 4.0) * 0.02)
            w_g += alpha_boost
            w_r -= alpha_boost * 0.5
            w_ml -= alpha_boost * 0.5

        # Clamp and normalize
        w_g = max(0.05, w_g)
        w_r = max(0.05, w_r)
        w_ml = max(0.05, w_ml)
        total = w_g + w_r + w_ml
        w_g /= total; w_r /= total; w_ml /= total
        
        # Weighted vote
        W = w_g * gamma_sig + w_r * regime_sig + w_ml * ml_sig

        # Threshold decision
        if W > self.THRESHOLD:
            sig = 1
        elif W < -self.THRESHOLD:
            sig = -1
        else:
            sig = 0

        strength = min(1.0, abs(W)) / (self.THRESHOLD + 1e-9) * 0.5
        return sig, strength
    

class MultiStrategyPipeline:

    def __init__(self) -> None:
        self._gamma = GammaSignalEngine(window=80, upper_pct=0.82, lower_pct=0.18)
        self._regime = RegimeClusterEngine(window=160, refit_every=30)
        self._supervised = SupervisedPredictionEngine(window=200, lookahead=4, refit_every=40)
        self._ensemble = MetaEnsemble()

    def process(self, event: MarketEvent) -> SignalEvent:
        gamma_sig = self._gamma.update(event)
        gamma_str = self._gamma.strength
        gamma_alph = self._gamma.fitted_alpha

        regime_id = self._regime.update(event)
        regime_sig = self._regime.regime_signal
        regime_stb = self._regime.stability

        ml_sig = self._supervised.update(event, regime_id)
        ml_conf = self._supervised.confidence

        final_sig, strength = self._ensemble.combine(
            gamma_sig=gamma_sig,
            gamma_strength=gamma_str,
            gamma_alpha=gamma_alph,
            regime_sig=regime_sig,
            regime_stab=regime_stb,
            ml_sig=ml_sig,
            ml_conf=ml_conf,
        )

        return SignalEvent(
            symbol=event.symbol,
            timestamp=event.timestamp,
            signal=final_sig,
            strength=strength,
            gamma_sig=gamma_sig,
            regime_id=regime_id,
            regime_sig=regime_sig,
            regime_stb=regime_stb,
            ml_sig=ml_sig,
            ml_conf=ml_conf,
        )
    
# --------------------------------------------------
# Risk Engine
# --------------------------------------------------

                
            
                








