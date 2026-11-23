import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import yfinance as yf


TICKER = "AAPL"
START_DATE = "2016-11-01"
END_DATE = "2018-11-01"

PORTFOLIO = 1_000_000.0
ALPHA = 0.97
BETA = 0.02
TRIALS = 125_000
SEED = 42


def download_close() -> pd.Series:
    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename("AAPL Close").dropna().astype(float)


def full_loss(log_returns: np.ndarray) -> np.ndarray:
    return PORTFOLIO * (1.0 - np.exp(log_returns))


def normal_full_var(mu: np.ndarray | float, sigma: np.ndarray | float):
    lower_return_quantile = norm.ppf(1.0 - ALPHA)
    return PORTFOLIO * (1.0 - np.exp(mu + sigma * lower_return_quantile))


prices = download_close()
returns = np.log(prices / prices.shift(1)).dropna().to_numpy()

n = returns.size
mean = float(returns.mean())
variance = float(returns.var(ddof=1))
standard_deviation = float(np.sqrt(variance))
degrees_freedom = n - 1

# Part (a): empirical estimate and fitted-normal plug-in estimate.
empirical_var = float(np.quantile(full_loss(returns), ALPHA))
theoretical_var = float(normal_full_var(mean, standard_deviation))

# With mu fixed, (n-1)S^2/sigma^2 ~ chi-square_(n-1).
sigma_lower = np.sqrt(
    degrees_freedom
    * variance
    / chi2.ppf(1.0 - BETA / 2.0, degrees_freedom)
)
sigma_upper = np.sqrt(
    degrees_freedom
    * variance
    / chi2.ppf(BETA / 2.0, degrees_freedom)
)
part_a_interval = np.asarray(
    normal_full_var(mean, np.array([sigma_lower, sigma_upper]))
)

# Part (b): invert the independent normal and chi-square pivots.
# W=(n-1)S^2/sigma^2 ~ chi-square_(n-1),
# Z=sqrt(n)(mu-Xbar)/sigma ~ N(0,1) (the negative of the usual pivot).
rng = np.random.default_rng(SEED)
w = rng.chisquare(degrees_freedom, size=TRIALS)
simulated_sigma = np.sqrt(degrees_freedom * variance / w)
z = rng.standard_normal(TRIALS)
simulated_mu = mean + simulated_sigma * z / np.sqrt(n)
simulated_var = np.asarray(normal_full_var(simulated_mu, simulated_sigma))

part_b_mean = float(simulated_var.mean())
part_b_interval = np.quantile(
    simulated_var, [BETA / 2.0, 1.0 - BETA / 2.0]
)

print("AAPL data")
print(f"Price observations: {len(prices)}")
print(f"Log returns (n):    {n}")
print(f"First date:         {prices.index[0].date()}")
print(f"Last date:          {prices.index[-1].date()}")
print(f"Sample mean:        {mean:.10f}")
print(f"Sample variance:    {variance:.10f}")
print(f"Sample std. dev.:   {standard_deviation:.10f}")

print("\nPart (a)")
print(f"Empirical VaR:           ${empirical_var:,.2f}")
print(f"Theoretical plug-in VaR: ${theoretical_var:,.2f}")
print(
    "98% CI, mu fixed:       "
    f"(${part_a_interval[0]:,.2f}, ${part_a_interval[1]:,.2f})"
)

print("\nPart (b)")
print(f"Trials:                   {TRIALS:,}")
print(f"Average simulated VaR:    ${part_b_mean:,.2f}")
print(f"98% empirical CI:        (${part_b_interval[0]:,.2f}, ${part_b_interval[1]:,.2f})")

print("\nComparison")
print(f"Empirical minus plug-in: ${empirical_var - theoretical_var:,.2f} ({empirical_var / theoretical_var - 1.0:.1%})")
    