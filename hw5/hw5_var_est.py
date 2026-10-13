import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf


tickers = ["MSFT", "AAPL", "GOOGL"]
start_date = "2024-05-23"
end_date = "2026-05-23"  # end is exclusive, so this includes 2026-05-22
alpha = .95
portfolio = 1_000_000

market_caps = {
    "MSFT": 3_109_000_000_000,
    "AAPL": 4_536_000_000_000,
    "GOOGL": 4_640_000_000_000,
}


# Task 1
price_series = {}
for ticker in tickers:
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    price_series[ticker] = close.rename(ticker)

prices = pd.concat(price_series.values(), axis=1).dropna()

print("\nTask 1: first five closing prices")
print(prices.head())
print("\nLast five closing prices")
print(prices.tail())
print(f"\nTotal observations: {len(prices)}")


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


returns = log_returns(prices)


# Task 2
weights = pd.Series(market_caps) / sum(market_caps.values())
weights = weights[tickers]

print("\nTask 2: market-cap weights")
print(weights)
print("Sum:", weights.sum())


# Task 3
def VaR(losses, alpha):
    return float(np.quantile(losses, alpha))


def full_loss(returns, weights, value):
    return value * (1 - np.exp(returns) @ weights)


def linear_loss(returns, weights, value):
    return -value * (returns @ weights)


def quadratic_loss(returns, weights, value):
    return -value * ((returns + .5 * returns**2) @ weights)


x = returns[tickers].to_numpy()
w = weights.to_numpy()

empirical_vars = {
    "full": VaR(full_loss(x, w, portfolio), alpha),
    "linear": VaR(linear_loss(x, w, portfolio), alpha),
    "quadratic": VaR(quadratic_loss(x, w, portfolio), alpha),
}

print("\nTask 3: empirical VaR")
for name, estimate in empirical_vars.items():
    print(f"{name:>9}: ${estimate:,.2f}")


# Task 4
lam = theta = .97


def ewma_mean_cov(observations, lam, theta):
    mu = np.zeros(observations.shape[1])
    covariance = np.zeros((observations.shape[1], observations.shape[1]))

    for row in observations:
        difference = row - mu
        covariance = theta * covariance + (1 - theta) * np.outer(difference, difference)
        mu = lam * mu + (1 - lam) * row

    return mu, covariance


mean, covariance = ewma_mean_cov(x, lam, theta)

print("\nTask 4: EWMA mean")
print(pd.Series(mean, index=tickers))
print("\nEWMA covariance")
print(pd.DataFrame(covariance, index=tickers, columns=tickers))


# Task 5
n = 100_000
rng = np.random.default_rng(42)
simulated_x = rng.multivariate_normal(mean, covariance, size=n)


def normal_linear_var(mean, covariance, weights, value, alpha):
    loss_mean = -value * weights @ mean
    loss_sd = value * np.sqrt(weights @ covariance @ weights)
    return loss_mean + loss_sd * norm.ppf(alpha)


normal_vars = {
    "full": VaR(full_loss(simulated_x, w, portfolio), alpha),
    "linear": normal_linear_var(mean, covariance, w, portfolio, alpha),
    "quadratic": VaR(quadratic_loss(simulated_x, w, portfolio), alpha),
}

print("\nTask 5: normal/EWMA VaR")
for name, estimate in normal_vars.items():
    print(f"{name:>9}: ${estimate:,.2f}")


# Task 6
comparison = pd.DataFrame({"empirical": empirical_vars, "normal/EWMA": normal_vars})
comparison["difference"] = comparison["normal/EWMA"] - comparison["empirical"]
comparison["percent difference"] = 100 * comparison["difference"] / comparison["empirical"]

print("\nTask 6: comparison")
print(comparison)

# Compare the full loss estimates first. A large gap is usually due to the
# normal model missing skew/heavy tails or EWMA putting more weight on recent volatility.

