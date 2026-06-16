"""
plot_greeks.py
==============
Visualise Black-Scholes call-option prices and Greeks (Delta, Gamma, Vega)
as functions of spot price S0 and time-to-maturity T.
 
Figures produced
----------------
figures/02_hockey_stick.png   — BS call price vs intrinsic value
figures/02_greeks_vs_S0.png   — Delta, Gamma, Vega as S0 varies (fixed T)
figures/02_greeks_vs_T.png    — Delta, Gamma, Vega as T varies (ATM, fixed S0=K)
 
Dependencies: numpy, matplotlib, and the project's own black_scholes module.
"""

from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np




from optpricing.black_scholes import bs_greeks, bs_price  # noqa: E402  (after sys.path tweak)


# Output directory
FIG_DIR = _HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# Global plot style
plt.rcParams.update({
    "figure.dpi":        110,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "semibold",
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "legend.frameon":    False,
    "mathtext.fontset":  "cm",
})


# Consistent colour palette used across all figures.
PALETTE = {
    "delta": "#1f77b4",  # blue
    "gamma": "#d62728",  # red
    "vega":  "#2ca02c",  # green
    "price": "#111111",  # near-black
    "ref":   "#888888",  # grey for reference / axis lines
}


# Utility
def stack_greeks(results: list[dict]) -> dict[str, np.ndarray]:
    """
    Transpose a list of per-point Greek dicts into a dict of arrays.
 
    Parameters
    ----------
    results : list of dict
        Each element is the output of bs_greeks() at one grid point,
        e.g. {'delta': 0.53, 'gamma': 0.019, 'vega': 19.4}.
 
    Returns
    -------
    dict[str, np.ndarray]
        Keys are Greek names; values are 1-D arrays over the grid.
 
    Example
    -------
    >>> stack_greeks([{'delta': 0.5, 'gamma': 0.02},
    ...               {'delta': 0.6, 'gamma': 0.01}])
    {'delta': array([0.5, 0.6]), 'gamma': array([0.02, 0.01])}
    """
    keys = results[0].keys()
    return {k: np.array([d[k] for d in results]) for k in keys}


# Figure 0 — Hockey-stick: BS call price vs intrinsic value
def plot_hockey_stick(
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
    s_lo: float = 60.0,
    s_hi: float = 140.0,
    n_pts: int = 300,
) -> None:
    """
    Plot the BS call price alongside the intrinsic (payoff) value.
 
    The "hockey stick" shape arises because:
      - For S0 >> K the option is deep in-the-money and priced near S0 - K.
      - For S0 << K the option is worthless at expiry but retains time value.
      - The smooth curve sits above the kinked payoff by exactly the time value.
    """
    S0_grid     = np.linspace(s_lo, s_hi, n_pts)
    call_prices = np.array([bs_price(s, K, T, r, sigma, "call") for s in S0_grid])
    intrinsic   = np.maximum(S0_grid - K, 0.0)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(S0_grid, call_prices, color=PALETTE["price"], lw=2,
        label=fr"BS call  ($T={T}$, $\sigma={sigma}$, $r={r}$)")
    ax.plot(S0_grid, intrinsic,   color=PALETTE["ref"],   lw=1.4, ls="--",
        label=r"Intrinsic value $\max(S_0 - K,\;0)$")
    ax.axvline(K, color=PALETTE["ref"], lw=0.8, ls=":")
    
    ax.set(
        xlabel=r"$S_0$",
        ylabel="Call price",
        title=fr"Black–Scholes call price vs spot  ($K={K:.0f}$)",
    )
    ax.legend(loc="upper left")
 
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_hockey_stick.png")
    print(f"Saved: {FIG_DIR / '02_hockey_stick.png'}")
    
    
# Figure 1 — Greeks vs spot S0  (fixed T, K, r, sigma)
def plot_greeks_vs_spot(
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
    s_lo: float = 60.0,
    s_hi: float = 140.0,
    n_pts: int = 300,
) -> None:
    """
    Plot Delta, Gamma, and Vega as functions of spot price S0.
 
    Analytical landmarks annotated on the figure
    ---------------------------------------------
    Delta ATM (S0 = K):
        d1 = [ln(S0/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
        At S0=K the log term is zero, leaving d1 = (r + sigma^2/2)*sqrt(T) / sigma > 0,
        so Delta_ATM = N(d1) > 0.5.  The excess above 1/2 reflects the positive
        risk-neutral drift built into geometric Brownian motion.
 
    Gamma peak location:
        Differentiating ln(Gamma) w.r.t. S0 and setting to zero yields
        d1 = -sigma*sqrt(T)  =>  S_peak = K * exp(-(r + 3*sigma^2/2)*T).
 
    Vega peak location:
        Vega = S0 * phi(d1) * sqrt(T).  The extra factor of S0^2 * sigma * T
        relative to Gamma shifts the peak:
        d1 = +sigma*sqrt(T)  =>  S_peak_vega = K * exp((sigma^2/2 - r)*T).
    """
    S0_grid = np.linspace(s_lo, s_hi, n_pts)
    greeks  = stack_greeks([bs_greeks(s, K, T, r, sigma, "call") for s in S0_grid])
    
    # Analytical landmark values
    delta_atm     = bs_greeks(K, K, T, r, sigma, "call")["delta"]
 
    S_gamma_peak  = K * np.exp(-(r + 1.5 * sigma**2) * T)
    gamma_peak    = bs_greeks(S_gamma_peak, K, T, r, sigma, "call")["gamma"]
 
    S_vega_peak   = K * np.exp((0.5 * sigma**2 - r) * T)
    
    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
 
    # Panel A: Delta
    ax = axes[0]
    ax.plot(S0_grid, greeks["delta"], color=PALETTE["delta"], lw=2)
    ax.axhline(0.5, color=PALETTE["ref"], lw=0.8, ls="--")
    ax.axvline(K,   color=PALETTE["ref"], lw=0.8, ls=":")
    ax.scatter([K], [delta_atm], color=PALETTE["delta"], zorder=5)
    ax.annotate(
        fr"$\Delta(K) = {delta_atm:.4f}$" "\n"
        r"$> \frac{1}{2}$ because $d_1$ contains" "\n"
        r"the drift $r + \frac{1}{2}\sigma^2$",
        xy=(K, delta_atm),
        xytext=(K + 7, 0.40),
        fontsize=9,
        color=PALETTE["delta"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["delta"], lw=0.8),
    )
    ax.set(xlabel=r"$S_0$", ylabel=r"$\Delta$", ylim=(0, 1),
           title="Delta vs spot")
 
    # Panel B: Gamma
    ax = axes[1]
    ax.plot(S0_grid, greeks["gamma"], color=PALETTE["gamma"], lw=2)
    ax.axvline(K,            color=PALETTE["ref"],   lw=0.8, ls=":")
    ax.axvline(S_gamma_peak, color=PALETTE["gamma"], lw=0.8, ls="--", alpha=0.6)
    ax.scatter([S_gamma_peak], [gamma_peak], color=PALETTE["gamma"], zorder=5)
    ax.annotate(
        "Gamma peak at\n"
        fr"$S^\star = K\,e^{{-(r + 3\sigma^2/2)\,T}} \approx {S_gamma_peak:.2f}$",
        xy=(S_gamma_peak, gamma_peak),
        xytext=(S_gamma_peak - 27, gamma_peak * 0.55),
        fontsize=9,
        color=PALETTE["gamma"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["gamma"], lw=0.8),
    )
    ax.set(xlabel=r"$S_0$", ylabel=r"$\Gamma$", title="Gamma vs spot")
 
    # Panel C: Vega
    ax = axes[2]
    ax.plot(S0_grid, greeks["vega"], color=PALETTE["vega"], lw=2)
    ax.axvline(K,           color=PALETTE["ref"],  lw=0.8, ls=":")
    ax.axvline(S_vega_peak, color=PALETTE["vega"], lw=0.8, ls="--", alpha=0.6)
    ax.text(
        0.04, 0.95,
        r"$\nu = S_0\,\varphi(d_1)\,\sqrt{T}$" "\n"
        r"$\Gamma = \varphi(d_1)\,/\,(S_0\,\sigma\sqrt{T})$" "\n"
        r"$\Rightarrow\;\nu = S_0^2\,\sigma\,T\,\Gamma$" "\n"
        fr"Vega peak: $S^\star_{{\nu}} \approx {S_vega_peak:.2f}$",
        transform=ax.transAxes,
        fontsize=8.5,
        color=PALETTE["vega"],
        va="top",
    )
    ax.set(xlabel=r"$S_0$", ylabel=r"$\nu$", title="Vega vs spot")
 
    fig.suptitle(
        fr"Greeks vs spot  ($K={K:.0f},\ T={T},\ r={r},\ \sigma={sigma}$)",
        fontweight="semibold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_greeks_vs_S0.png")
    print(f"Saved: {FIG_DIR / '02_greeks_vs_S0.png'}")


# Figure 2 — Greeks vs maturity T  (ATM: S0 = K, fixed r, sigma)
 
def plot_greeks_vs_maturity(
    S0: float = 100.0,
    K: float  = 100.0,
    r: float  = 0.05,
    sigma: float = 0.20,
    t_lo: float  = 0.05,   # must be > 0: Gamma ~ 1/sqrt(T) diverges at T=0
    t_hi: float  = 10.0,   # extended so the Vega interior maximum is visible
    n_pts: int   = 400,
) -> None:
    """
    Plot Delta, Gamma, and Vega as functions of time-to-maturity T (ATM).
 
    Analytical behaviour
    --------------------
    At-the-money (S0 = K), define beta = (r + sigma^2/2) / sigma.
    Then d1 = beta * sqrt(T).
 
    Delta ATM:
        Delta = N(d1) = N(beta * sqrt(T)).
        Monotonically increasing from 0.5 at T=0 toward 1 as T->inf.
        Intuition: a longer-dated ATM call has more time for the stock to
        drift above K under the risk-neutral measure (drift = r > 0).
 
    Gamma ATM:
        Gamma proportional to  exp(-beta^2 * T / 2) / sqrt(T).
        The 1/sqrt(T) factor dominates near zero, causing the blow-up visible
        on the left of the plot.  As T->0, all of the payoff's curvature
        concentrates at the kink S0=K (a Dirac delta in the limit).
 
    Vega ATM:
        Vega = S0 * phi(beta * sqrt(T)) * sqrt(T)
             = S0 * exp(-beta^2 * T / 2) / sqrt(2*pi) * sqrt(T).
        Two competing effects:
          - sqrt(T) -> 0 as T -> 0    (no time to move => no vol exposure)
          - exp(-beta^2*T/2) -> 0 as T -> inf  (d1 grows large => phi -> 0)
        These balance at T* = 1 / beta^2 (the interior Vega maximum).
        For r=0.05, sigma=0.20: beta = 0.35, T* = 1/0.35^2 ~ 8.16 yr.
        The grid extends to 10 yr specifically to reveal this peak and the
        subsequent decay — this is a visualisation choice, not a trading one.
    """
    T_grid = np.linspace(t_lo, t_hi, n_pts)
    greeks = stack_greeks([bs_greeks(S0, K, t, r, sigma, "call") for t in T_grid])
 
    # Vega interior maximum: T* = 1 / beta^2
    beta       = (r + 0.5 * sigma**2) / sigma
    T_star     = 1.0 / beta**2
    vega_peak  = bs_greeks(S0, K, T_star, r, sigma, "call")["vega"]
 
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
 
    # Panel A: Delta vs T
    # Delta_ATM = N(beta * sqrt(T)) increases monotonically with T.
    ax = axes[0]
    ax.plot(T_grid, greeks["delta"], color=PALETTE["delta"], lw=2)
    ax.axhline(0.5, color=PALETTE["ref"], lw=0.8, ls="--")
    ax.set(xlabel=r"$T$ (years)", ylabel=r"$\Delta$",
           title=r"Delta vs maturity (ATM, $S_0 = K$)")
    ax.text(
        0.04, 0.95,
        r"$\Delta_\mathrm{ATM} = N\!\left(\beta\sqrt{T}\right)$" "\n"
        r"$\beta = (r + \frac{1}{2}\sigma^2)\,/\,\sigma$" "\n"
        "Monotone in $T$: positive drift\n"
        r"raises $d_1$ as maturity grows.",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["delta"],
        va="top",
    )
 
    # Panel B: Gamma vs T
    # Gamma_ATM ~ exp(-beta^2*T/2) / sqrt(T) — diverges as T -> 0.
    ax = axes[1]
    ax.plot(T_grid, greeks["gamma"], color=PALETTE["gamma"], lw=2)
    ax.set(xlabel=r"$T$ (years)", ylabel=r"$\Gamma$",
           title="Gamma vs maturity (ATM)")
    ax.text(
        0.45, 0.95,
        r"$\Gamma_\mathrm{ATM} \propto \dfrac{e^{-\beta^2 T/2}}{\sqrt{T}}$" "\n\n"
        r"As $T\to 0$: $\Gamma\to\infty$." "\n"
        "Payoff kink at $S_0 = K$\n"
        "concentrates all curvature.",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["gamma"],
        va="top",
    )
 
    # Panel C: Vega vs T
    # Interior maximum at T* = 1/beta^2 due to competing sqrt(T) and phi(d1) terms.
    ax = axes[2]
    ax.plot(T_grid, greeks["vega"], color=PALETTE["vega"], lw=2)
    ax.axvline(T_star, color=PALETTE["vega"], lw=0.8, ls="--", alpha=0.6)
    ax.scatter([T_star], [vega_peak], color=PALETTE["vega"], zorder=5)
    ax.annotate(
        fr"$T^\star = 1/\beta^2 \approx {T_star:.2f}$ yr" "\n"
        r"$\beta = (r + \frac{1}{2}\sigma^2)\,/\,\sigma$",
        xy=(T_star, vega_peak),
        xytext=(T_star + 0.8, vega_peak * 0.55),
        fontsize=9,
        color=PALETTE["vega"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["vega"], lw=0.8),
    )
    ax.set(xlabel=r"$T$ (years)", ylabel=r"$\nu$",
           title="Vega vs maturity (ATM)")
 
    fig.suptitle(
        fr"Greeks vs maturity  ($S_0 = K = {K:.0f},\ r={r},\ \sigma={sigma}$)",
        fontweight="semibold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_greeks_vs_T.png")
    print(f"Saved: {FIG_DIR / '02_greeks_vs_T.png'}")
    
    
    
def main() -> None:
    plot_hockey_stick()
    plot_greeks_vs_spot()
    plot_greeks_vs_maturity()
    plt.show()
 
 
if __name__ == "__main__":
    main()