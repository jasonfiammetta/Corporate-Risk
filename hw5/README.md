# HW5 Starter Guide

The three VaR scripts are complete, runnable versions in a casual homework style.
The ES file remains a written-answer skeleton with short derivation hints.

- `hw5_var_est.py` covers all six questions in `05_var_est_hw1_pub.pdf`.
- `hw5_var_backtest.py` covers all four tasks in `05_var_backtest_hw_pub.pdf`.
- `hw5_var_10day.py` covers all three tasks in `05_var_10day_hw_pub.pdf`.
- `hw5_es_skeleton.md` gives a derivation path for all five questions (and every
  subpart) in `06_es_hw_pub.pdf`.
- `hw5_var_backtest_results.csv` is generated when the backtest runs.
- `risk_utils.py` is an optional reference containing reusable versions of the
  common functions; the homework scripts themselves are self-contained.

## Running the Python work

From this directory:

```bash
python -m pip install -r requirements.txt
python hw5_var_est.py
python hw5_var_backtest.py
python hw5_var_10day.py
```

The first two scripts need network access and `yfinance`; the 10-day option script
is self-contained.  All random simulations use fixed seeds for reproducibility.


## Conventions worth stating in the submitted work

1. Yahoo Finance's `end` parameter is exclusive, so the scripts use 2026-05-23
   to include the requested 2026-05-22 close.
2. Historical and backtest losses use the full loss convention
   \(L=-(V_{t+1}-V_t)\), so a positive value is a loss.
3. A VaR forecast uses only information available before the realized return it
   is tested against.
4. The 10-day PDF describes a short **put**, although one sentence labels
   \(P^{BS}\) a call price.  The starter follows the short-put wording and uses a
   Black-Scholes put price and put delta.
5. Task 4 of the ES sheet asks for strict VaR monotonicity, but that claim is
   false without more assumptions; the notes provide a counterexample.

## Submission checklist

- Replace the `TODO` prompts in the ES notes with your own concise conclusions.
- Include units and the confidence level next to every reported VaR/ES.
- State the data observation count and date range actually returned.
- For Monte Carlo answers, state the number of paths and random seed.
- If your class uses a different empirical quantile convention, pass the intended
  method to `numpy.quantile` consistently and mention it.
