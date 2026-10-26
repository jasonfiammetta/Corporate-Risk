import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

# download initial data
msft_data = yf.download(
    "MSFT",
    start="2011-09-01",
    end="2016-09-02", # include 09-01
    interval="1d",
    auto_adjust=True,
)
aapl_data = yf.download(
    "AAPL",
    start="2011-09-01",
    end="2016-09-02", # include 09-01
    interval="1d",
    auto_adjust=True,
)

msft = msft_data['Close'].iloc[:, 0]
aapl = aapl_data['Close'].iloc[:, 0]

msft_returns = np.log(msft / msft.shift(1)).dropna()
aapl_returns = np.log(aapl / aapl.shift(1)).dropna()

returns = np.array((msft_returns, aapl_returns))

# setup vars
alpha = 0.95
k = 10
portfolio_value = 1_000_000
lam = theta = 0.97
m = 50_000

# 1. Estimating EWMA with recursive equations
def ewma(returns, lam, theta):
    d, _ = returns.shape

    mu = np.zeros(d)
    Sigma = np.zeros((d, d))

    for t, x in enumerate(returns.T):
        l = x - mu # compute Sigma using previous mu before update
        mu = lam * mu + (1 - lam) * x
        Sigma = theta * Sigma + (1 - theta) * np.outer(l, l)

    return mu, Sigma

mu_t, Sigma_t = ewma(returns, lam, theta)
print("1.")
print("mu:", mu_t)
print("Sigma:\n", Sigma_t)

# 2. Estimate VaR of linear loss
print("\n2.")
market_caps = np.array([448_770_000_000, 575_110_000_000])
weights = market_caps / (market_caps.sum())

# calculate loss using derived formulas
theta_t = portfolio_value * weights
L_lin = -theta_t @ mu_t.T
L_lin_sd = np.sqrt(theta_t.T @ Sigma_t @ theta_t)

print(f"L_lin:         $ {L_lin:,.2f}")
print(f"L_lin_sd:      $  {L_lin_sd:.2f}\n")

VaR_1day = L_lin + norm.ppf(alpha) * L_lin_sd
VaR_10day = np.sqrt(10) * VaR_1day
regulatory_capital = 3 * VaR_10day

print(f"1 day VaR:                   $  {VaR_1day:,.2f}")
print(f"10 day VaR:                  $  {VaR_10day:,.2f}")
print(f"Regulatory Capital required: $ {regulatory_capital:,.2f}")

# 3 Shock the system
print("\n3.")
# EWMA standard deviations
sig_msft = np.sqrt(Sigma_t[0, 0])
sig_aapl = np.sqrt(Sigma_t[1, 1])

# correction shock for aapl returns
x_aapl = mu_t[1] - 5 * sig_aapl

# EWMA correlation
rho = Sigma_t[0, 1] / (sig_msft * sig_aapl)
rho = np.clip(rho, -1.0, 1.0)

# conditional distribution of msft given the aapl shock
tilde_mu_msft = mu_t[0] + rho * (sig_msft / sig_aapl) * (x_aapl - mu_t[1])
tilde_var_msft = Sigma_t[0, 0] * (1 - rho**2)
tilde_sd_msft = np.sqrt(max(tilde_var_msft, 0.0))

# another deterministic five-sigma shock
x_msft = tilde_mu_msft - 5 * tilde_sd_msft

# Day-one shocked return vector
x_shock = np.array([x_msft, x_aapl])
print(f"shocked returns: {x_shock}")

shock_residual = x_shock - mu_t

mu_t_2delta = lam * mu_t + (1 - lam) * x_shock
Sigma_t_2delta = theta * Sigma_t + (1 - theta) * np.outer(shock_residual, shock_residual)

# 4. 
print("\n4.")

losses = np.empty(m)

for path in range(m):
    # keep original first day
    mu_path = mu_t_2delta.copy()
    Sigma_path = Sigma_t_2delta.copy()

    # Day one is the deterministic shock
    cumulative_return = x_shock.copy()

    # Simulate days 2 through K
    for day in range(2, k + 1):
        x = np.random.multivariate_normal(mu_path, Sigma_path)
        cumulative_return += x

        # No update is needed after the final day
        if day >= k:
            break
        
        residual = x - mu_path

        # get next day's values
        mu_next = lam * mu_path + (1 - lam) * x
        Sigma_next = theta * Sigma_path + (1 - theta) * np.outer(residual, residual)

        mu_path = mu_next
        Sigma_path = Sigma_next

    losses[path] = -theta_t @ cumulative_return

print(f"simulated {m} paths")

print("\n5.")
average_loss = losses.mean()
shocked_VaR_10day = np.quantile(losses, alpha)

var_exceedance_pct = 100 * np.mean(losses > VaR_10day)
capital_exceedance_pct = 100 * np.mean(losses > regulatory_capital)


print(f"  (a) average K-day portfolio loss:   $ {average_loss:,.2f}")
print(f"  (b) estimate of K-day VaR:          $ {shocked_VaR_10day:,.2f}")
print(f"  (c) frequency of exceedences                        {var_exceedance_pct:.2f}%")
print(f"  (d) frequency of exceedences of regulatory capital:  {capital_exceedance_pct:.2f}%")



