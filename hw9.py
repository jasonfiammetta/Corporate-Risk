from itertools import product
from math import exp

r = 0.0132
T = 0.25
delta = 1 / 252
t0 = 0.0
t5 = 5 * delta

S1_0, S2_0 = 158.12, 170.33
sigma1_0, sigma2_0 = 0.2214, 0.3036
rho_0 = 0.237

X_VALUES = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)
VOL_MULTIPLIERS = (0.50, 0.75, 1.25, 1.50, 1.75, 2.00)
RHO_SHIFTS = (-0.50, 0.0, 0.50)

# Definition for unhedged portfolio value (see part a)
def portfolio_value(t, s1, s2, sig1, sig2, rho):
    tau = T - t
    return exp(r * tau) * (
        s1 ** 2 * exp(sig1 ** 2 * tau)
        - 2 * s1 * s2 * exp(sig1 * sig2 * rho * tau)
        + s2 ** 2 * exp(sig2 ** 2 * tau)
    )

# definition for hts
def deltas(t, s1, s2, sig1, sig2, rho):
    tau = T - t
    common = exp(r * tau)
    cross = exp(sig1 * sig2 * rho * tau)
    h1 = 2 * common * (s1 * exp(sig1**2 * tau) - s2 * cross)
    h2 = 2 * common * (s2 * exp(sig2**2 * tau) - s1 * cross)
    return h1, h2


# Initial portolio and fixed hedge
V = portfolio_value(t0, S1_0, S2_0, sigma1_0, sigma2_0, rho_0)
h1_0, h2_0 = deltas(t0, S1_0, S2_0, sigma1_0, sigma2_0, rho_0)
# Initial hedged portfolio value
V_0 = V - h1_0 * S1_0 - h2_0 * S2_0

rows = []

# loop through all combinations
for x1, x2, m1, m2, drho in product(
    X_VALUES, X_VALUES, VOL_MULTIPLIERS, VOL_MULTIPLIERS, RHO_SHIFTS
):
    # Get new prices at t+5 for given combo
    S1_5 = S1_0 * exp(x1)
    S2_5 = S2_0 * exp(x2)
    sigma1_1 = sigma1_0 * m1
    sigma2_1 = sigma2_0 * m2
    rho_1 = rho_0 + drho

    V_5 = portfolio_value(t5, S1_5, S2_5, sigma1_1, sigma2_1, rho_1)

    # use initial hedge values to adjust to hedged porfolio
    V_5 -= (h1_0 * S1_5 + h2_0 * S2_5)
    loss = V_0 - V_5

    rows.append(
        {
            "x1": x1,
            "x2": x2,
            "m1": m1,
            "m2": m2,
            "drho": drho,
            "loss": loss,
        }
    )

assert len(rows) == 3888

# Part (b) answer
worst = max(rows, key=lambda row: row["loss"])
x1, x2, m1, m2, drho, loss = worst.values()
print(f"The worst-case scenario risk measure loss is {loss} at {x1=}, {x2=}, {m1=}, {m2=}, {drho=}")

# Part (c)
def x_weight(x):
    return {0.20: 0.50, 0.10: 0.75, 0.05: 1.00}[round(abs(x), 2)]

def vol_weight(multiplier):
    return {
        0.50: 0.50,
        0.75: 1.25,
        1.25: 0.75,
        1.50: 0.75,
        1.75: 1.25,
        2.00: 0.50,
    }[multiplier]

def rho_weight(shift):
    return 0.50 if shift == 0.0 else 1.25

rows.clear()

# loop through all combinations
for x1, x2, m1, m2, drho in product(
    X_VALUES, X_VALUES, VOL_MULTIPLIERS, VOL_MULTIPLIERS, RHO_SHIFTS
):
    # Get new prices at t+5 for given combo
    S1_5 = S1_0 * exp(x1)
    S2_5 = S2_0 * exp(x2)
    sigma1_1 = sigma1_0 * m1
    sigma2_1 = sigma2_0 * m2
    rho_1 = rho_0 + drho

    V_5 = portfolio_value(t5, S1_5, S2_5, sigma1_1, sigma2_1, rho_1)

    # use initial hedge values to adjust to hedged porfolio
    V_5 -= (h1_0 * S1_5 + h2_0 * S2_5)
    loss = V_0 - V_5

    weight = (
        x_weight(x1)
        * x_weight(x2)
        * vol_weight(m1)
        * vol_weight(m2)
        * rho_weight(drho)
    )

    rows.append(
        {
            "x1": x1,
            "x2": x2,
            "m1": m1,
            "m2": m2,
            "drho": drho,
            "weighted_loss": weight * loss,
        }
    )

assert len(rows) == 3888

# Part (c) answer
weighted_worst = max(rows, key=lambda row: row["weighted_loss"])
x1, x2, m1, m2, drho, weighted_loss = weighted_worst.values()
print(f"The worst-case scenario risk measure weighted loss is {weighted_loss} at {x1=}, {x2=}, {m1=}, {m2=}, {drho=}")

