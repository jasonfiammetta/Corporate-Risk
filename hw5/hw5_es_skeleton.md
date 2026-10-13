# Expected Shortfall Homework Skeleton

Use the loss convention throughout: if the question gives a change in portfolio
value, set `loss = -change in value`.  Define

\[
\operatorname{VaR}_\alpha(L)=\inf\{\ell:F_L(\ell)\geq\alpha\},\qquad
\operatorname{ES}_\alpha(L)=\frac{1}{1-\alpha}\int_\alpha^1
\operatorname{VaR}_u(L)\,du.
\]

The integral definition matters for discrete distributions.  Simply averaging all
observations satisfying \(L\geq\operatorname{VaR}\) can give the wrong ES when
there is probability mass at the VaR.

## 1. Normal 10-day loss

Because a centered normal change and its negative have the same distribution,
\(L\sim N(0,20^2)\) in millions.  Let \(z_{0.99}=\Phi^{-1}(0.99)\).

\[
\operatorname{VaR}_{0.99}(L)=20z_{0.99},\qquad
\operatorname{ES}_{0.99}(L)=20\frac{\phi(z_{0.99})}{1-0.99}.
\]

TODO: evaluate both expressions, retain units of millions of dollars, and state
that ES must exceed VaR.

## 2. Two independent binary losses

First write the convolution table:

| Loss | \(P(L_1)\) | \(P(L_2)\) | \(P(L_1+L_2)\) |
|---:|---:|---:|---:|
| 0 | 0.98 | 0.98 | \(0.98^2\) |
| 10 | 0.02 | 0.02 | \(2(0.98)(0.02)\) |
| 20 | 0 | 0 | \(0.02^2\) |

### 2(a): \(\operatorname{VaR}_{0.98}\)

- For each individual loss, inspect \(F_{L_i}(0)\).
- For the sum, compare 0.98 with \(F_{L_1+L_2}(0)\) and
  \(F_{L_1+L_2}(10)\).
- Insert the three quantiles into
  \(\operatorname{VaR}(L_1+L_2)\leq
  \operatorname{VaR}(L_1)+\operatorname{VaR}(L_2)\).

TODO: state the numerical inequality that fails.

### 2(b): \(\operatorname{VaR}_{0.99}\)

Repeat the same CDF lookup at 0.99.  The sum does not need to have strictly
smaller VaR for subadditivity; it only needs to satisfy the weak inequality.

### 2(c): \(\operatorname{ES}_{0.98}\)

For each \(L_i\), the worst 2% is exactly the loss-10 state.  For the sum, build
the worst 2% from the top down:

1. include all probability \(0.02^2\) at loss 20;
2. fill the remaining tail probability with just enough of the loss-10 atom;
3. divide the probability-weighted tail loss by 0.02.

TODO: compute all three ES values and verify
\(\operatorname{ES}(L_1+L_2)\leq \operatorname{ES}(L_1)+\operatorname{ES}(L_2)\).

## 3. Two bank loans

Work in millions of dollars.  A performing loan has loss \(-0.2\); a defaulted
loan has loss \(U\sim\operatorname{Unif}(0,10)\).

### 3(a): one-loan 99% VaR

The marginal default probability of Loan 1 is 0.0125, so

\[
P(L_1=-0.2)=0.9875.
\]

The 99th percentile lies inside the default component.  Convert it to a
conditional uniform percentile:

\[
q=\frac{0.99-0.9875}{0.0125},\qquad
\operatorname{VaR}_{0.99}(L_1)=F_U^{-1}(q)=10q.
\]

TODO: evaluate \(q\) and the VaR.

### 3(b): portfolio 99% VaR

- Neither defaults (probability 0.975): portfolio loss is \(-0.4\).
- Exactly one defaults (probability 0.025): portfolio loss is \(U-0.2\).

Thus the conditional percentile and portfolio VaR are

\[
q_P=\frac{0.99-0.975}{0.025},\qquad
\operatorname{VaR}_{0.99}(L_1+L_2)=10q_P-0.2.
\]

TODO: compare this with twice the answer from 3(a) and state whether VaR is
subadditive.

### 3(c): one-loan and portfolio ES

The worst 1% is contained entirely in the continuous default component in both
cases.  After finding each VaR cutoff, use the mean of a uniform upper interval:

\[
E[U\mid U>c]=\frac{c+10}{2}.
\]

For the portfolio, remember the performing loan contributes \(-0.2\) to loss.
TODO: calculate both ES values and check that portfolio ES is subadditive.

## 4. Claimed strict monotonicity

The requested statement is **not true in general** for VaR without extra
assumptions.  Strict pointwise ordering gives weak quantile ordering, but not
necessarily strict ordering at every confidence level.

A compact counterexample: let \(U\sim\operatorname{Unif}(0,1)\),

\[
L_1=U,\qquad L_2=U+(U-1/2)^2.
\]

Then \(L_1<L_2\) almost surely (equality occurs only at \(U=1/2\), a probability
zero event).  The map defining \(L_2\) is increasing on \([0,1]\), so its
quantiles are obtained by applying that map to the quantiles of \(U\).  At
\(\alpha=1/2\), fill in both VaRs and observe that they are equal.

TODO: present the counterexample cleanly, then say that the universally valid
conclusion is only
\(\operatorname{VaR}_\alpha(L_1)\leq\operatorname{VaR}_\alpha(L_2)\).
If the instructor intended a proof, ask whether an additional condition (for
example, a deterministic positive gap) was omitted.

## 5. Exponential and Pareto losses

A useful continuous-tail identity is

\[
\operatorname{ES}_\alpha(L)=E[L\mid L>\operatorname{VaR}_\alpha(L)].
\]

### 5(a): exponential mean \(\theta\)

Solve \(1-e^{-q/\theta}=\alpha\) for \(q\).  Then use exponential
memorylessness:

\[
E[L\mid L>q]=q+\theta.
\]

TODO: substitute the quantile into the ES expression.

### 5(b): Pareto unit scale and tail parameter \(\xi\)

From \(1-q^{-1/\xi}=\alpha\), solve for the VaR.  For ES, either integrate the
quantile from \(\alpha\) to 1 or use the Pareto conditional mean.

Key checkpoint: the mean of this Pareto distribution is finite only when the
shape \(1/\xi>1\), equivalently \(0<\xi<1\).  In that case ES is a constant
multiple of VaR; when \(\xi\geq1\), ES is infinite even though VaR is finite.

TODO: derive the multiplier and state the two \(\xi\) cases explicitly.

