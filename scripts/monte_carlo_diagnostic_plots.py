"""
Diagnostics for the Monte-Carlo Black-Scholes option pricer.
 
This module produces visual + numerical checks that the MC engine is:
  (a) converging to the analytic price at the theoretical O(N^-1/2) rate,
  (b) unbiased (the sampling distribution centres on the Black-Scholes price),
  (c) producing well-calibrated 95% confidence intervals (~95% coverage).
 
It is *diagnostic* code: it consumes the pricer, it does not test it. The unit
tests for `mc_price` live in the pytest suite. Nothing here should ever be the
thing that proves correctness on its own.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from optpricing.monte_carlo import mc_price
from optpricing.black_scholes import bs_price

Z_975:float = float(norm.ppf(0.975))

# Market parameters bundled into one immutable object
@dataclass(frozen=True)
class MarketParams:
    """
    A single option contract / market state.
 
    `frozen=True` makes instances hashable and prevents a function from
    accidentally mutating the shared parameters mid-experiment.
    """
    S0: float = 100.0          # spot price
    K: float = 100.0           # strike
    T: float = 1.0             # time to maturity (years)
    r: float = 0.05            # risk-free rate (annualised, continuous)
    sigma: float = 0.20        # volatility (annualised)
    option_type: str = "call"  # "call" or "put"
    
    
def _mc(params: MarketParams, n_paths: int, seed:int | None) -> dict:
    """Thin adapter so the rest of the file never positionally unpacks args."""
    return mc_price(
        params.S0, params.K, params.T, params.r, params.sigma,
        params.option_type, n_paths, seed=seed,    
    )


def _bs(params: MarketParams) -> float:
    """Analytic Black-Scholes price for the same contract."""
    return bs_price(
        params.S0, params.K, params.T, params.r, params.sigma,
        params.option_type,
    )


def _independent_seeds(master_seed: int, size: int) -> np.ndarray:
    """
    Return `size` independent integer seeds derived from one master seed.
 
    Using a master RNG to spawn child seeds gives reproducibility (fix the
    master) *and* independence (each draw is its own stream)
    
    Collisions are astronomically unlikely in a 2**32 space
    for the M we use.
    """
    rng = np.random.default_rng(master_seed)
    return rng.integers(0, 2**32, size=size)



# Convergence of the price estimate as N grows
def plot_convergence(
    params: MarketParams,
    bs_true: float,
    *,
    n_min_exp: int = 2,
    n_max_exp: int = 6,
    n_points: int = 20,
    master_seed: int = 0,
    ax: plt.Axes | None = None,
):    
    """
    Plot the MC price (with 95% band) against path count N.
 
    Returns (fig, N_grid, prices, ses) so the SE-scaling plot can reuse the
    exact same measured standard errors instead of recomputing them.
    """
    # Geometric grid in N. .astype(int) + np.unique collapses the duplicate
    # small integers you get when log-spaced points round to the same value.
    N_grid = np.unique(np.logspace(n_min_exp, n_max_exp, n_points).astype(int))
    seeds = _independent_seeds(master_seed, len(N_grid))
    
    prices = np.empty(len(N_grid))
    ses = np.empty(len(N_grid))
    for i, (N,s) in enumerate(zip(N_grid, seeds)):
        res = _mc(params, n_paths=int(N), seed=int(s))
        prices[i], ses[i] = res["price"], res["std_error"]
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
 
    ax.semilogx(N_grid, prices, "o-", color="C0", ms=4, label="MC estimate")
    ax.fill_between(
        N_grid,
        prices - Z_975 * ses,
        prices + Z_975 * ses,
        alpha=0.25, color="C0", label="95% CI band",
    )
    ax.axhline(bs_true, color="k", ls="--", lw=1.2, label="Black-Scholes")
    ax.set_xlabel("number of paths $N$")
    ax.set_ylabel("option price")
    ax.set_title("MC price convergence")
    ax.legend(frameon=False)
 
    # VERIFY: the band half-width is Z_975 * SE ~ N^-1/2, so it should roughly
    # halve every time N quadruples. We just print the endpoints as a sanity ratio.
    print(
        f"[convergence] band half-width: {Z_975 * ses[0]:.4f} (N={N_grid[0]}) "
        f"-> {Z_975 * ses[-1]:.4f} (N={N_grid[-1]})"
    )
    return fig, N_grid, prices, ses
 
 

# Standard-error scaling: SE ~ N^-1/2
def plot_se_scaling(
    N_grid: np.ndarray,
    ses: np.ndarray,
    *,
    ax: plt.Axes | None = None,
):
    """
    Log-log plot of standard error vs N, with a -1/2 reference slope.
 
    Returns (fig, fitted_slope). Reuses the (N_grid, ses) measured upstream so
    the two figures are guaranteed consistent.
    """
    N_grid = np.asarray(N_grid, dtype=float)
    ses = np.asarray(ses, dtype=float)
 
    # Edge case: logs blow up on non-positive SE. 
    if np.any(ses <= 0):
        raise ValueError("all standard errors must be positive to take logs")
 
    # Reference line anchored at the first point: SE(N) = SE(N0) * sqrt(N0 / N).
    se_ref = ses[0] * np.sqrt(N_grid[0] / N_grid)
 
    # Fit slope of log(SE) vs log(N). Theory says exactly -1/2.
    slope = np.polyfit(np.log(N_grid), np.log(ses), 1)[0]
    print(f"[se_scaling] fitted slope = {slope:.3f}   (expected ~ -0.500)")
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
 
    ax.loglog(N_grid, ses, "o", color="C1", label="measured SE")
    ax.loglog(N_grid, se_ref, "--", color="k", lw=1.2,
              label=r"reference slope 1/2")
    ax.set_xlabel("number of paths $N$")
    ax.set_ylabel("standard error of the estimator")
    ax.set_title(f"SE scaling (fit slope = {slope:.3f})")
    ax.legend(frameon=False)
    return fig, slope
 
 
# Sampling distribution of the estimator
def plot_sampling_distribution(
    params: MarketParams,
    bs_true: float,
    *,
    N: int,
    M: int,
    master_seed: int = 0,
    ax: plt.Axes | None = None,
):
    """
    Histogram of M independent MC estimates, each from N paths.
 
    The CLT says these estimates are ~ Normal(bs_true, SE^2). We overlay that
    normal density. Returns (fig, mean, std, se_theory).
    """
    # PITFALL FIX: independence *requires distinct seeds*. One shared seed would
    # give M identical numbers and a degenerate single-bar "histogram".
    seeds = _independent_seeds(master_seed, M)
 
    est = np.empty(M)
    reported_se = np.empty(M)
    for j, s in enumerate(seeds):
        res = _mc(params, n_paths=N, seed=int(s))
        est[j] = res["price"]
        reported_se[j] = res["std_error"]
 
    # Theoretical scale = the SE the engine reports (averaged over runs). This
    # is the engine's own self-consistency check: does its reported SE match
    # the *observed* spread of estimates?
    se_theory = float(reported_se.mean())
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
 
    ax.hist(est, bins=40, density=True, alpha=0.55, color="C2",
            label="MC estimates")
    x = np.linspace(est.min(), est.max(), 400)
    ax.plot(x, norm.pdf(x, loc=bs_true, scale=se_theory), "k-", lw=1.6,
            label="CLT prediction")
    ax.axvline(bs_true, color="k", ls="--", lw=1.0, label="Black-Scholes")
    ax.set_xlabel("MC price estimate")
    ax.set_ylabel("density")
    ax.set_title(f"Sampling distribution (N={N}, M={M})")
    ax.legend(frameon=False)
 
    mean, std = float(est.mean()), float(est.std(ddof=1))
    print(
        f"[sampling] mean={mean:.4f} (bs_true={bs_true:.4f}), "
        f"observed std={std:.4f}, reported se_theory={se_theory:.4f}"
    )
    return fig, mean, std, se_theory
 
 
# Confidence-interval coverage
def plot_ci_coverage(
    params: MarketParams,
    bs_true: float,
    *,
    N: int,
    M: int,
    master_seed: int = 1,
    max_intervals: int = 100,
    ax: plt.Axes | None = None,
):
    """
    Run M independent MC pricings and check how often each 95% CI covers
    the true price. A correctly calibrated engine covers ~95% of the time.
 
    Returns (fig, coverage).
    """
    seeds = _independent_seeds(master_seed, M)
 
    los = np.empty(M)
    his = np.empty(M)
    for j, s in enumerate(seeds):
        res = _mc(params, n_paths=N, seed=int(s))
        los[j], his[j] = res["ci_95"]
 
    contains = (los <= bs_true) & (his >= bs_true)   # bool array, shape (M,)
    coverage = float(contains.mean())
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
 
    # Plotting all M intervals gets unreadable for large M, so cap the display
    # (the coverage number above still uses all M runs).
    n_show = min(M, max_intervals)
    idx = np.arange(n_show)
    hit, miss = contains[:n_show], ~contains[:n_show]
 
    # Two vlines calls -> clean two-entry legend (green = covers, red = misses).
    ax.vlines(idx[hit], los[:n_show][hit], his[:n_show][hit],
              color="C2", lw=1.0, label="covers truth")
    ax.vlines(idx[miss], los[:n_show][miss], his[:n_show][miss],
              color="C3", lw=1.0, label="misses truth")
    ax.axhline(bs_true, color="k", ls="--", lw=1.2, label="Black-Scholes")
    ax.set_xlabel(f"run index (showing {n_show} of {M})")
    ax.set_ylabel("price")
    ax.set_title(f"95% CI coverage = {coverage:.3f}")
    ax.legend(frameon=False)
 
    # VERIFY: coverage ~ [0.92, 0.98] for M=200. That band is p +/- 2*SE_p with
    # SE_p = sqrt(p(1-p)/M) = sqrt(0.95*0.05/200) ~ 0.0154, i.e. the finite-M
    # noise on the coverage estimate itself.
    print(f"[ci_coverage] empirical coverage = {coverage:.3f}  (target 0.95)")
    return fig, coverage
 
 
# Strike/type comparison table with the z-score diagnostic
def comparison_table(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    strikes,
    n_paths: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    MC-vs-BS table across strikes and option types.
 
    z = (MC - BS) / SE is the real diagnostic: under correct pricing it is
    ~ N(0,1) for each row, so |z| < 1.96 should hold for ~95% of rows.
    """
    rows = []
    for K in strikes:
        for opt in ("call", "put"):
            res = mc_price(S0, K, T, r, sigma, opt, n_paths, seed=seed)
            bs = bs_price(S0, K, T, r, sigma, opt)
            err = res["price"] - bs
            lo, hi = res["ci_95"]
            rows.append({
                "K": K,
                "type": opt,
                "BS": bs,
                "MC": res["price"],
                "SE": res["std_error"],
                "error": err,
                "z": err / res["std_error"],
                "in_CI": lo <= bs <= hi,
            })
    return pd.DataFrame(rows)
 
 
def main(*, show: bool = True, save: bool = True, outdir: str = ".") -> None:
    """
    Run all diagnostics.
 
    show=True   -> render the figures (pop-up window, or Spyder's Plots pane).
    save=True   -> also write each figure to <outdir> as a PNG.
    outdir      -> where the PNGs go; created if missing. Default = current
                   working directory (which under Spyder is usually wherever
                   the script lives).
    """
    params = MarketParams()              # S0=100, K=100, T=1, r=0.05, sigma=0.20, call
    bs_true = _bs(params)
    print(f"Black-Scholes reference price = {bs_true:.6f}\n")
 
    # Collect (figure, filename) so saving/showing/closing is handled uniformly.
    figures: list[tuple[plt.Figure, str]] = []
 
    # 1 + 2 share data: compute the convergence sweep once, reuse the SEs.
    fig1, N_grid, _prices, ses = plot_convergence(params, bs_true)
    figures.append((fig1, "mc_convergence.png"))
 
    fig2, _slope = plot_se_scaling(N_grid, ses)
    figures.append((fig2, "mc_se_scaling.png"))
 
    fig3, *_ = plot_sampling_distribution(params, bs_true, N=20_000, M=500)
    figures.append((fig3, "mc_sampling_distribution.png"))
 
    fig4, _coverage = plot_ci_coverage(params, bs_true, N=20_000, M=500)
    figures.append((fig4, "mc_ci_coverage.png"))
 
    if save:
        out = Path(outdir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)   # no-op if it already exists
        for fig, name in figures:
            path = out / name
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"saved {path}")               # absolute path, easy to find
 
    table = comparison_table(
        S0=params.S0, T=params.T, r=params.r, sigma=params.sigma,
        strikes=[80, 90, 100, 110, 120], n_paths=50_000,
    )
    # to_string avoids pandas truncating columns in the console.
    print("\n" + table.round(4).to_string(index=False))
 
    if show:
        plt.show()                               # renders everything at once
    else:
        for fig, _ in figures:                   # batch mode: free the memory
            plt.close(fig)
 
 
if __name__ == "__main__":
    # Interactive default: see the plots AND keep the PNGs.
    main(show=True, save=True)