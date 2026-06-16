import numpy as np
import pytest

from optpricing.black_scholes import bs_price

def _parity_residual(S0, K, T, r, sigma):
    """C - P - (S0 - K·e^{-rT}) — should be zero."""
    C = bs_price(S0, K, T, r, sigma, "call")
    P = bs_price(S0, K, T, r, sigma, "put")
    return (C - P) - (S0 - K * np.exp(-r * T))

# parametrize runs the same test across every row automatically
@pytest.mark.parametrize("S0, K, T, r, sigma", [
    (100, 100, 1.0,  0.05, 0.20),
    (80,  100, 0.5,  0.02, 0.30),
    (120,  90, 2.0,  0.04, 0.15),
    (100, 100, 0.01, 0.05, 0.20),   # near-expiry
    (100, 100, 1.0,  0.05, 0.01),   # near-zero vol
])
def test_put_call_parity(S0, K, T, r, sigma):
    assert abs(_parity_residual(S0, K, T, r, sigma)) < 1e-8

def test_parity_vectorised():
    """Parity should hold across a grid of spot prices."""
    S0s = np.linspace(60, 140, 50)
    residuals = _parity_residual(S0s, K=100, T=1.0, r=0.05, sigma=0.20)
    assert np.all(np.abs(residuals) < 1e-8)