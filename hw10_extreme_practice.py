import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import genextreme, genpareto, norm


portfolio = 1_000_000.0
lam = theta = 0.97
alpha = 0.95

alpha_range = np.linspace(0.99, 0.9999, 100)

# download S&P 500 data
data = yf.download(
    "^GSPC",
    start="2011-11-01",
    end="2021-12-29",
    interval="1d",
    auto_adjust=False,
    progress=False,
)
close = data["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
prices = close.rename("close").dropna()

# 1. Emperical distribution for log returns
returns = np.log(prices / prices.shift(1)).dropna()
losses = (-portfolio * returns).to_numpy()
empirical_var = np.quantile(losses, alpha_range)

print(f"Price observations: {len(prices)}")
print(f"Log returns:        {len(returns)}")

# 2. EWMA estimates of mean and variance
def ewma_forecast(returns: pd.Series, as_of: str = "2021-11-01") -> tuple[float, float]:
    history = returns.loc[:as_of]

    initial = history.iloc[:100]
    mu = float(initial.mean())
    sigma2 = float(initial.var(ddof=1))

    # iterated theta*sigma + (1-theta)*delta^2
    for x in history.iloc[100:]:
        delta = float(x) - mu
        sigma2 = theta * sigma2 + (1.0 - theta) * delta**2
        mu = lam * mu + (1.0 - lam) * float(x)

    return mu, sigma2

# loss mean=-V*mu ; loss sd=V*sigma
mu, sigma2 = ewma_forecast(returns)
normal_var = -portfolio * mu + portfolio * np.sqrt(sigma2) * norm.ppf(alpha_range)

print(f"\nEWMA forecast through 2021-11-01: mu={mu:.8f}, sigma={np.sqrt(sigma2):.8f}")

# 3. GEV distrbution
def gev_daily_var(losses: np.ndarray, block_size: int, alpha_range: np.ndarray):
    maxima = losses.reshape(-1, block_size).max(axis=1)
    # Fit loss fractions rather than large dollar values for numerical stability.
    c, loc_fraction, scale_fraction = genextreme.fit(maxima / portfolio)

    # If H_n(x) ~= F_L(x)^n, then F_L(x)=alpha corresponds to H_n(x)=alpha^n.
    daily_var = portfolio * genextreme.ppf(
        alpha_range**block_size, c=c, loc=loc_fraction, scale=scale_fraction
    )
    return daily_var, maxima, (c, portfolio * loc_fraction, portfolio * scale_fraction)


def gp_daily_var(losses: np.ndarray, alpha_range: np.ndarray):
    threshold = float(np.quantile(losses, alpha))
    excesses = losses[losses > threshold] - threshold
    xi, loc, beta = genpareto.fit(excesses, floc=0.0)

    # Conditional excess probability corresponding to the unconditional alpha.
    conditional_p = (alpha_range - alpha) / (1.0 - alpha)
    daily_var = threshold + genpareto.ppf(
        conditional_p, c=xi, loc=0.0, scale=beta
    )
    return daily_var, threshold, excesses, (xi, beta)

gev_73_var, maxima_73, gev_73 = gev_daily_var(losses, 73, alpha_range)
gev_35_var, maxima_35, gev_35 = gev_daily_var(losses, 35, alpha_range)
gp_var, threshold, excesses, gp_params = gp_daily_var(losses, alpha_range)

print(f"GEV n=73: scipy c={gev_73[0]:.6f}, EVT xi={-gev_73[0]:.6f}, "
        f"loc={gev_73[1]:.2f}, scale={gev_73[2]:.2f}, maxima={len(maxima_73)}")
print(f"GEV n=35: scipy c={gev_35[0]:.6f}, EVT xi={-gev_35[0]:.6f}, "
        f"loc={gev_35[1]:.2f}, scale={gev_35[2]:.2f}, maxima={len(maxima_35)}")
print(f"GP: u={threshold:.2f}, xi={gp_params[0]:.6f}, beta={gp_params[1]:.2f}, "
        f"exceedances={len(excesses)}")


# 4. GP excess loss distrubition
curves = pd.DataFrame(
    {
        "alpha": alpha_range,
        "empirical": empirical_var,
        "normal_ewma": normal_var,
        "gev_n73_m35": gev_73_var,
        "gev_n35_m73": gev_35_var,
        "gp_u95": gp_var,
    }
)

print("\nVaR at alpha=.9999")
print(curves.iloc[-1].drop(labels="alpha").map(lambda x: f"${x:,.2f}"))
highest = curves.columns[1:][np.argmax(curves.iloc[-1, 1:].to_numpy())]
print(f"\nHighest at alpha=.9999: {highest}")
print("Absolute difference between the two GEV estimates: "f"${abs(gev_73_var[-1] - gev_35_var[-1]):,.2f}")

# curves.to_csv("hw10_var_curves.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 6))
for column in curves.columns[1:]:
    ax.plot(curves["alpha"], curves[column], label=column)
ax.set(
    title="HW 10: one-day S&P 500 loss VaR",
    xlabel=r"Confidence level $\alpha$",
    ylabel="VaR ($)",
)
ax.ticklabel_format(style="plain", axis="y")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig("hw10_var_curves.png", dpi=180)
plt.close(fig)

