import numpy as np
import pytest
from optpricing.black_scholes import bs_price, bs_greeks

# Baseline at-the-money option fixture
# S0=K means we are exactly ATM, which is where Greeks are
# most sensitive and most interesting to test.
BASE = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.20)

# Numerical gradient helpers
# These use the central-difference formula with O(h^2) truncuation error
def _numerical_delta(opt, h=0.01, S0=None):
    """
    Central-difference estimate of Delta

    Parameters
    ----------
    opt : str
        'call' or 'put'
    h : float
        Step size in spot. The default is 0.01.
    S0 : float, optional
         Spot price at which to evaluate delta. Defaults to BASE['S0']
    """
    p = BASE.copy()
    s = S0 if S0 is not None else p["S0"]
    return (bs_price(s + h, p["K"], p["T"], p["r"], p["sigma"], opt)
          - bs_price(s - h, p["K"], p["T"], p["r"], p["sigma"], opt)) / (2 * h)

def _numerical_vega(h=0.001):
    """
    Central difference estimate of Vega for a call.
    
    Vega is identical for calls and puts (put-call parity), so testing
    the call is sufficient.
    """
    p = BASE.copy()
    return (bs_price(p["S0"], p["K"], p["T"], p["r"], p["sigma"] + h, "call")
          - bs_price(p["S0"], p["K"], p["T"], p["r"], p["sigma"] - h, "call")) / (2 * h)

def _numerical_gamma(opt, h=0.5):
    """
    Central-difference estimate of Gamma
    
    Gamma is approximated by differencing delta at S0+h and S0-h. 
    This is a second-order finite difference of a first-order finite
    difference, so the total truncation error is still O(h^2), but with
    a larger prefactor than for delta alone.
    """
    p = BASE.copy()
    delta_up   = _numerical_delta(opt, S0=p["S0"] + h)
    delta_down = _numerical_delta(opt, S0=p["S0"] - h)
    return (delta_up - delta_down) / (2 * h)

def _numerical_theta(opt, h=1/365):
    """
    Central-difference estimate of Theta = dV/dt (calendar-time convention).
 
    Note the sign: Theta is dV/dt where t is calendar time, so it is
    dV/d(-T). 
    """
    p = BASE.copy()
    return -(bs_price(p["S0"], p["K"], p["T"] + h, p["r"], p["sigma"], opt)
           - bs_price(p["S0"], p["K"], p["T"] - h, p["r"], p["sigma"], opt)) / (2 * h)
 
 
def _numerical_rho(opt, h=0.001):
    """Central-difference estimate of Rho = dV/dr."""
    p = BASE.copy()
    return (bs_price(p["S0"], p["K"], p["T"], p["r"] + h, p["sigma"], opt)
          - bs_price(p["S0"], p["K"], p["T"], p["r"] - h, p["sigma"], opt)) / (2 * h)
    


# Tests 

@pytest.mark.parametrize("opt", ["call", "put"])
def test_delta_matches_numerical(opt):
    """
    Analytic delta Phi(d1) [call] or Phi(d1)-1 [put] matches dV/dS0
    via central difference. Tolerance 1e-5 is consistent with O(h^2)
    truncation at h=0.01
    """
    analytic  = bs_greeks(**BASE, option_type=opt)["delta"]
    numerical = _numerical_delta(opt)
    assert abs(analytic - numerical) < 1e-5, (
        f"{opt} delta: analytic={analytic:.8f}, numerical={numerical:.8f}"
    )


def test_vega_matches_numerical():
    """
    Analytic vega S0*phi(d1)*sqrt(T) matches dV/d(sigma) via central
    difference. Vega is identical for calls and puts (put-call parity).
    """
    analytic  = bs_greeks(**BASE, option_type="call")["vega"]
    numerical = _numerical_vega()
    assert abs(analytic - numerical) < 1e-4, (
    f"vega: analytic={analytic:.8f}, numerical={numerical:.8f}"
    )



@pytest.mark.parametrize("opt", ["call", "put"])
def test_gamma_matches_numerical(opt):
    """
    Analytic gamma phi(d1)/(S0*sigma*sqrt(T)) matches d^2V/dS0^² via
    second-order central difference. Tolerance 1e-4 is appropriate for
    a second derivative with h=0.5.
    """
    analytic  = bs_greeks(**BASE, option_type=opt)["gamma"]
    numerical = _numerical_gamma(opt)
    assert abs(analytic - numerical) < 1e-4, (
        f"{opt} gamma: analytic={analytic:.8f}, numerical={numerical:.8f}"
    )
 
 
@pytest.mark.parametrize("opt", ["call", "put"])
def test_theta_matches_numerical(opt):
    """
    Analytic theta matches -dV/dT via central difference.
    Tolerance is looser (1e-3) because theta involves a product of several
    terms and h=1/365 (one trading day) is already fairly small.
    """
    analytic  = bs_greeks(**BASE, option_type=opt)["theta"]
    numerical = _numerical_theta(opt)
    assert abs(analytic - numerical) < 1e-3, (
        f"{opt} theta: analytic={analytic:.8f}, numerical={numerical:.8f}"
    )
 
 
@pytest.mark.parametrize("opt", ["call", "put"])
def test_rho_matches_numerical(opt):
    """Analytic rho matches dV/dr via central difference."""
    analytic  = bs_greeks(**BASE, option_type=opt)["rho"]
    numerical = _numerical_rho(opt)
    assert abs(analytic - numerical) < 1e-4, (
        f"{opt} rho: analytic={analytic:.8f}, numerical={numerical:.8f}"
    )
 
 
def test_delta_put_call_symmetry():
    """
    Delta put-call parity: Delta_call - Delta_put = 1.
    """
    d_call = bs_greeks(**BASE, option_type="call")["delta"]
    d_put  = bs_greeks(**BASE, option_type="put")["delta"]
    assert abs(d_call - d_put - 1.0) < 1e-10, (
        f"delta parity violated: d_call - d_put = {d_call - d_put:.10f}"
    )
 
 
def test_gamma_identical_call_put():
    """
    Gamma is identical for calls and puts.
    """
    g_call = bs_greeks(**BASE, option_type="call")["gamma"]
    g_put  = bs_greeks(**BASE, option_type="put")["gamma"]
    assert abs(g_call - g_put) < 1e-10
 
 
def test_vega_identical_call_put():
    """
    Vega is identical for calls and puts.
    """
    v_call = bs_greeks(**BASE, option_type="call")["vega"]
    v_put  = bs_greeks(**BASE, option_type="put")["vega"]
    assert abs(v_call - v_put) < 1e-10
 
 
def test_put_call_parity_price():
    """
    Price-level put-call parity: C - P = S0 - K*exp(-rT).
 
    This is a model-free no-arbitrage identity. If this fails, the
    pricing function itself is inconsistent.
    """
    c = bs_price(**BASE, option_type="call")
    p = bs_price(**BASE, option_type="put")
    parity_rhs = BASE["S0"] - BASE["K"] * np.exp(-BASE["r"] * BASE["T"])
    assert abs((c - p) - parity_rhs) < 1e-10, (
        f"Put-call parity violated: C-P={c-p:.10f}, S-K*disc={parity_rhs:.10f}"
    )
 
 
def test_delta_boundary():
    """
    Deep ITM call: Delta -> 1 (N(d1) -> 1 as S0/K -> inf).
    Deep OTM call: Delta -> 0 (N(d1) -> 0 as S0/K -> 0).
    """
    deep_itm = bs_greeks(S0=200, K=100, T=1.0, r=0.05, sigma=0.20,
                         option_type="call")["delta"]
    deep_otm = bs_greeks(S0=50,  K=100, T=1.0, r=0.05, sigma=0.20,
                         option_type="call")["delta"]
    assert deep_itm > 0.999
    assert deep_otm < 0.001
 
 
def test_greeks_at_expiry_itm():
    """
    At T=0 with S0 > K (ITM call): Delta=1, Gamma=Vega=0.
    The option is certain to be exercised; it behaves like long stock.
    """
    g = bs_greeks(S0=110, K=100, T=0.0, r=0.05, sigma=0.20, option_type="call")
    assert g["delta"] == 1.0
    assert g["gamma"] == 0.0
    assert g["vega"]  == 0.0
 
 
def test_greeks_at_expiry_otm():
    """
    At T=0 with S0 < K (OTM call): Delta=0, Gamma=Vega=0.
    The option expires worthless; no sensitivity to any input.
    """
    g = bs_greeks(S0=90, K=100, T=0.0, r=0.05, sigma=0.20, option_type="call")
    assert g["delta"] == 0.0
    assert g["gamma"] == 0.0
    assert g["vega"]  == 0.0
 
 
def test_greeks_zero_vol():
    """
    At sigma=0, the asset evolves deterministically to F = S0*exp(r*T).
    For an ITM call (F > K): Delta=1, Gamma=Vega=0.
    Rho and Theta reflect the discounted intrinsic value of the forward.
    """
    g = bs_greeks(S0=110, K=100, T=1.0, r=0.05, sigma=0.0, option_type="call")
    assert g["gamma"] == 0.0
    assert g["vega"]  == 0.0
    assert g["delta"] == 1.0
    
    
@pytest.mark.parametrize("kwargs", [
    dict(S0=-1,  K=100, T=1.0, r=0.05, sigma=0.20),
    dict(S0=100, K=0,   T=1.0, r=0.05, sigma=0.20),
    dict(S0=100, K=100, T=-1,  r=0.05, sigma=0.20),
    dict(S0=100, K=100, T=1.0, r=0.05, sigma=-0.1),
])
def test_input_validation_raises(kwargs):
    with pytest.raises(ValueError):
        bs_greeks(**kwargs, option_type="call")