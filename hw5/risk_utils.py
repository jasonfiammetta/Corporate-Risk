"""Small, reusable risk-measure helpers for the HW5 exercises.

The functions here contain the mechanical parts of the homework.  The scripts in
this directory still leave the interpretation and written discussion to you.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom, norm


FloatArray = NDArray[np.float64]


def close_prices(download: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """Return a consistently shaped DataFrame of closing prices from yfinance.

    yfinance returns different column layouts for one ticker and many tickers;
    this helper hides that difference and aligns all assets on common dates.
    """
    if download.empty:
        raise ValueError("The download returned no observations.")

    if isinstance(download.columns, pd.MultiIndex):
        level0 = download.columns.get_level_values(0)
        if "Close" not in level0:
            raise KeyError("No 'Close' field in the yfinance result.")
        prices = download["Close"].copy()
    else:
        if "Close" not in download.columns:
            raise KeyError("No 'Close' field in the yfinance result.")
        prices = download[["Close"]].copy()
        if tickers and len(tickers) == 1:
            prices.columns = tickers

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    if tickers is not None and set(tickers).issubset(prices.columns):
        prices = prices[tickers]
    return prices.dropna(how="any").astype(float)


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Compute aligned log returns, dropping the first missing row."""
    return np.log(prices / prices.shift(1)).dropna(how="any")


def var(losses: ArrayLike, alpha: float = 0.95) -> float:
    """Empirical VaR using NumPy's default linearly interpolated quantile."""
    values = np.asarray(losses, dtype=float)
    if values.size == 0:
        raise ValueError("VaR requires at least one loss observation.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    return float(np.quantile(values, alpha))


def expected_shortfall(losses: ArrayLike, alpha: float = 0.95) -> float:
    """Empirical ES as the average of the worst (1-alpha) fraction.

    Fractional weight at the boundary makes this agree with the integral-of-VaR
    definition even when the sample has atoms or (1-alpha)*n is not an integer.
    """
    values = np.sort(np.asarray(losses, dtype=float).ravel())[::-1]
    if values.size == 0:
        raise ValueError("ES requires at least one loss observation.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    tail_count = (1.0 - alpha) * values.size
    whole = int(np.floor(tail_count))
    fraction = tail_count - whole
    tail_sum = values[:whole].sum()
    if fraction > 0.0:
        tail_sum += fraction * values[whole]
    return float(tail_sum / tail_count)


def full_loss(log_return_scenarios: ArrayLike, weights: ArrayLike, value: float) -> FloatArray:
    """Full revaluation loss V*[1 - sum_i w_i exp(X_i)]."""
    x = np.atleast_2d(np.asarray(log_return_scenarios, dtype=float))
    w = np.asarray(weights, dtype=float)
    return value * (1.0 - np.exp(x) @ w)


def linear_loss(log_return_scenarios: ArrayLike, weights: ArrayLike, value: float) -> FloatArray:
    """First-order loss approximation -V*w'X."""
    x = np.atleast_2d(np.asarray(log_return_scenarios, dtype=float))
    w = np.asarray(weights, dtype=float)
    return -value * (x @ w)


def quadratic_loss(log_return_scenarios: ArrayLike, weights: ArrayLike, value: float) -> FloatArray:
    """Second-order Taylor loss -V*sum_i w_i*(X_i + X_i^2/2)."""
    x = np.atleast_2d(np.asarray(log_return_scenarios, dtype=float))
    w = np.asarray(weights, dtype=float)
    return -value * ((x + 0.5 * x**2) @ w)


@dataclass(frozen=True)
class EwmaEstimate:
    mean: FloatArray
    covariance: FloatArray


def ewma_mean_cov(
    observations: ArrayLike,
    mean_decay: float = 0.97,
    covariance_decay: float = 0.97,
    initial_mean: ArrayLike | None = None,
    initial_covariance: ArrayLike | None = None,
) -> EwmaEstimate:
    """Apply the exact EWMA recursion printed in the assignment.

    The covariance innovation uses the *old* mean, mu_t, before mu is updated.
    """
    x = np.atleast_2d(np.asarray(observations, dtype=float))
    dimension = x.shape[1]
    mu = np.zeros(dimension) if initial_mean is None else np.asarray(initial_mean, dtype=float).copy()
    sigma = (
        np.zeros((dimension, dimension))
        if initial_covariance is None
        else np.asarray(initial_covariance, dtype=float).copy()
    )

    for row in x:
        innovation = row - mu
        sigma = covariance_decay * sigma + (1.0 - covariance_decay) * np.outer(innovation, innovation)
        mu = mean_decay * mu + (1.0 - mean_decay) * row
    return EwmaEstimate(mu, sigma)


def normal_linear_var(
    mean: ArrayLike,
    covariance: ArrayLike,
    weights: ArrayLike,
    value: float,
    alpha: float = 0.95,
) -> float:
    """Exact VaR when X is normal and L=-V*w'X."""
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    w = np.asarray(weights, dtype=float)
    loss_mean = -value * float(w @ mu)
    loss_sd = value * float(np.sqrt(max(w @ sigma @ w, 0.0)))
    return loss_mean + loss_sd * float(norm.ppf(alpha))


def exceedance_count_interval(periods: int, alpha: float = 0.95, beta: float = 0.05) -> tuple[int, int]:
    """Equal-tailed (1-beta) binomial interval for the exceedance count."""
    probability = 1.0 - alpha
    lower = int(binom.ppf(beta / 2.0, periods, probability))
    upper = int(binom.ppf(1.0 - beta / 2.0, periods, probability))
    return lower, upper


def black_scholes_put_price(
    s0: ArrayLike,
    k: float,
    T: float,
    r: float,
    sigma: float,
) -> FloatArray:
    """Black-Scholes European put price, vectorized over s0."""
    s = np.asarray(s0, dtype=float)
    if T <= 0.0:
        return np.maximum(k - s, 0.0)
    root_t = np.sqrt(T)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * T) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    return k * np.exp(-r * T) * norm.cdf(-d2) - s * norm.cdf(-d1)


def black_scholes_put_delta(
    s0: float,
    k: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Delta dP/dS of a European put."""
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )
    return float(norm.cdf(d1) - 1.0)


def gbm_terminal_spots(
    s0: float,
    mu: float,
    sigma: float,
    horizon: float,
    samples: int,
) -> FloatArray:
    """Sample exact GBM terminal prices under the physical measure."""
    shocks = np.random.standard_normal(samples)
    exponent = (mu - 0.5 * sigma**2) * horizon + sigma * np.sqrt(horizon) * shocks
    return s0 * np.exp(exponent)


def delta_hedged_short_put_losses(
    terminal_spots: ArrayLike,
    s0: float,
    k: float,
    T: float,
    r: float,
    sigma: float,
    horizon: float,
    contracts: int = 100,
) -> FloatArray:
    """Losses for the handout's fixed-delta short-put portfolio.

    This follows the equations in the PDF: Y_t=0 before the hedge trade,
    h=dP/dS, Y_new=-h*S, and the hedge is held fixed over the horizon.
    """
    if horizon > T:
        raise ValueError("The risk horizon cannot exceed option T.")
    terminal = np.asarray(terminal_spots, dtype=float)
    put_0 = float(black_scholes_put_price(s0, k, T, r, sigma))
    delta_0 = black_scholes_put_delta(s0, k, T, r, sigma)
    cash_0 = -delta_0 * s0
    portfolio_0 = -put_0  # cash_0 + delta_0*s0 cancel by construction
    put_h = black_scholes_put_price(terminal, k, T - horizon, r, sigma)
    portfolio_h = delta_0 * terminal + cash_0 * np.exp(r * horizon) - put_h
    return contracts * (portfolio_0 - portfolio_h)
