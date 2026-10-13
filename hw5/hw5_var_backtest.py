import numpy as np
import pandas as pd
from scipy.stats import binom, norm
import yfinance as yf


alpha = .95
beta = .05
lam = theta = .97
window = 1010


# Task 1
data = yf.download(
    "^GSPC",
    start="2016-05-23",
    end="2026-05-23",
    interval="1d",
    auto_adjust=False,
    progress=False,
)

spx = data["Close"]
if isinstance(spx, pd.DataFrame):
    spx = spx.iloc[:, 0]
spx = spx.rename("close").dropna()

print("\nTask 1: first five closes")
print(spx.head())
print("\nLast five closes")
print(spx.tail())
print(f"\nPrice observations: {len(spx)}")

returns = np.log(spx / spx.shift(1)).dropna()
losses = 1 - np.exp(returns)
n = len(returns)
j = n - window

print(f"Log-return observations: {n}")
print(f"Evaluation periods: {j}")


# Task 2
historical_var = []
for i in range(window, n):
    window_losses = losses.iloc[i - window:i]
    historical_var.append(np.quantile(window_losses, alpha))

dates = returns.index[window:]
realized_losses = losses.iloc[window:].to_numpy()
historical_var = np.array(historical_var)
historical_exceedance = realized_losses > historical_var


# Task 3
mu_hat = returns.iloc[:window].mean()
sigma2_hat = returns.iloc[:window].var(ddof=1)
normal_var = []

# find exceedances
for observed_return in returns.iloc[window:]:
    estimate = 1 - np.exp(mu_hat + np.sqrt(sigma2_hat) * norm.ppf(1 - alpha))
    normal_var.append(estimate)

    difference = observed_return - mu_hat
    sigma2_hat = theta * sigma2_hat + (1 - theta) * difference**2
    mu_hat = lam * mu_hat + (1 - lam) * observed_return

normal_var = np.array(normal_var)
normal_exceedance = realized_losses > normal_var

results = pd.DataFrame(
    {
        "loss": realized_losses,
        "historical_var": historical_var,
        "historical_exceedance": historical_exceedance,
        "normal_ewma_var": normal_var,
        "normal_ewma_exceedance": normal_exceedance,
    },
    index=dates,
)

print("\nTasks 2 and 3: first and last five estimates")
print(pd.concat([results.head(), results.tail()]))


# Task 4
lower = int(binom.ppf(beta / 2, j, 1 - alpha))
upper = int(binom.ppf(1 - beta / 2, j, 1 - alpha))

historical_count = int(historical_exceedance.sum())
normal_count = int(normal_exceedance.sum())

print(f"\nTask 4: {(1-beta):.0%} exceedance-count interval: [{lower}, {upper}]")
print(f"Historical exceedances: {historical_count} ({lower <= historical_count <= upper})")
print(f"Normal/EWMA exceedances: {normal_count} ({lower <= normal_count <= upper})")

results.to_csv("hw5_var_backtest_results.csv")
