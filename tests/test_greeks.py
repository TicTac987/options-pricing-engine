import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from black_scholes import bs_price, bs_greeks

BASE = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.20)

# ── Numerical gradient helpers ────────────────────────────────────────────────

def _numerical_delta(opt, h=0.01):
    p = BASE.copy()
    return (bs_price(p["S0"]+h, p["K"], p["T"], p["r"], p["sigma"], opt)
          - bs_price(p["S0"]-h, p["K"], p["T"], p["r"], p["sigma"], opt)) / (2*h)

def _numerical_vega(h=0.001):
    p = BASE.copy()
    return (bs_price(p["S0"], p["K"], p["T"], p["r"], p["sigma"]+h, "call")
          - bs_price(p["S0"], p["K"], p["T"], p["r"], p["sigma"]-h, "call")) / (2*h)

# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("opt", ["call", "put"])
def test_delta_matches_numerical(opt):
    analytic  = bs_greeks(**BASE, option_type=opt)["delta"]
    numerical = _numerical_delta(opt)
    assert abs(analytic - numerical) < 1e-5, f"{opt} delta mismatch"

def test_vega_matches_numerical():
    analytic  = bs_greeks(**BASE, option_type="call")["vega"]
    numerical = _numerical_vega()
    assert abs(analytic - numerical) < 1e-4

def test_delta_put_call_symmetry():
    """Δ_call − Δ_put = 1 (put-call parity)."""
    d_call = bs_greeks(**BASE, option_type="call")["delta"]
    d_put  = bs_greeks(**BASE, option_type="put")["delta"]
    assert abs(d_call - d_put - 1.0) < 1e-10

def test_gamma_identical_call_put():
    g_call = bs_greeks(**BASE, option_type="call")["gamma"]
    g_put  = bs_greeks(**BASE, option_type="put")["gamma"]
    assert abs(g_call - g_put) < 1e-10

def test_delta_boundary():
    deep_itm = bs_greeks(S0=200, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")["delta"]
    deep_otm = bs_greeks(S0=50,  K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")["delta"]
    assert deep_itm > 0.999
    assert deep_otm < 0.001

@pytest.mark.parametrize("kwargs", [
    dict(S0=-1,  K=100, T=1.0, r=0.05, sigma=0.20),
    dict(S0=100, K=0,   T=1.0, r=0.05, sigma=0.20),
    dict(S0=100, K=100, T=-1,  r=0.05, sigma=0.20),
    dict(S0=100, K=100, T=1.0, r=0.05, sigma=-0.1),
])
def test_input_validation_raises(kwargs):
    with pytest.raises(ValueError):
        bs_greeks(**kwargs, option_type="call")