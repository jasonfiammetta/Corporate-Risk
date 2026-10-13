import numpy as np
from scipy.stats import norm


alpha = .95
option_maturity = .292
dt = 1 / 252
days = 10
s0 = 152.51
k = 170
r = .00119
sigma = .4907
mu = .1691
n = 40_000
puts = 100


def gbm(s0, mu, sigma, horizon, n, rng):
    z = rng.standard_normal(n)
    return s0 * np.exp((mu - .5 * sigma**2) * horizon + sigma * np.sqrt(horizon) * z)


def black_scholes_put_price(spot, k, maturity, r, sigma):
    if maturity <= 0:
        return np.maximum(k - spot, 0)
    d1 = (np.log(spot / k) + (r + .5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return k * np.exp(-r * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def black_scholes_put_delta(spot, k, maturity, r, sigma):
    d1 = (np.log(spot / k) + (r + .5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    return norm.cdf(d1) - 1


def simulated_VaR(horizon_days, rng):
    horizon = horizon_days * dt
    terminal_spots = gbm(s0, mu, sigma, horizon, n, rng)

    put_0 = black_scholes_put_price(s0, k, option_maturity, r, sigma)
    h = black_scholes_put_delta(s0, k, option_maturity, r, sigma)
    cash = -h * s0
    value_0 = -put_0

    put_1 = black_scholes_put_price(
        terminal_spots, k, option_maturity - horizon, r, sigma
    )
    value_1 = h * terminal_spots + cash * np.exp(r * horizon) - put_1
    losses = puts * (value_0 - value_1)

    return np.quantile(losses, alpha)


# Task 1
one_day_var = simulated_VaR(1, np.random.default_rng(42))
sqrt_time_var = np.sqrt(days) * one_day_var

# Task 2
direct_ten_day_var = simulated_VaR(days, np.random.default_rng(43))

print(f"One-day VaR:                ${one_day_var:,.2f}")
print(f"10-day VaR (sqrt-time):     ${sqrt_time_var:,.2f}")
print(f"10-day VaR (direct):        ${direct_ten_day_var:,.2f}")
print(f"Difference:                 ${direct_ten_day_var - sqrt_time_var:,.2f}")
print(f"Percent difference:          {(direct_ten_day_var / sqrt_time_var - 1):.1%}")

# Task 3: the direct estimate can differ because option value is nonlinear in
# the stock price and time to maturity changes. The square-root rule assumes a
# much more linear, i.i.d. loss process.

