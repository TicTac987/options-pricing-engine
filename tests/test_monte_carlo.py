import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monte_carlo import mc_price
from black_scholes import bs_price


@pytest.mark.slow
def test_convergence_to_bs():
    S0 = K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    N = 1_000_000

    mc = mc_price(S0, K, T, r, sigma, "call", N, seed=123)
    bs = bs_price(S0, K, T, r, sigma, "call")

    assert abs(mc["price"] - bs) < 4 * mc["std_error"], (
        f"mc_price={mc['price']:.6f}, bs_price={bs:.6f}, "
        f"std_error={mc['std_error']:.6f}"
    )


@pytest.mark.slow
def test_ci_coverage():
    S0 = K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    true_price = bs_price(S0, K, T, r, sigma, "call")

    M = 200
    N = 50_000

    hits = 0
    for seed in range(M):
        res = mc_price(S0, K, T, r, sigma, "call", N, seed=seed)
        lo, hi = res["ci_95"]
        if lo <= true_price <= hi:
            hits += 1

    coverage = hits / M

    assert 0.905 <= coverage <= 0.985, (
        f"coverage {coverage:.3f} outside binomial band "
        f"[0.905, 0.985] with M={M}"
    )


@pytest.mark.slow
def test_se_scaling():
    S0 = K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    Ns = np.array([1_000, 10_000, 100_000, 1_000_000])
    seeds = [1, 2, 3, 4, 5]
    ses = []

    for N in Ns:
        se_vals = []
        for seed in seeds:
            res = mc_price(S0, K, T, r, sigma, "call", N, seed=seed)
            se_vals.append(res["std_error"])
        ses.append(np.mean(se_vals))

    ses = np.array(ses)

    logN = np.log(Ns)
    logSE = np.log(ses)
    slope, intercept = np.polyfit(logN, logSE, 1)

    expected = -0.5
    tol = 0.03
    err = abs(slope - expected)

    assert err < tol, (
        "SE scaling test failed:\n"
        f"  estimated slope: {slope:.5f}\n"
        f"  expected slope: {expected}\n"
        f"  error: {err:.5f}\n"
        f"  Ns: {Ns.tolist()}\n"
        f"  SEs: {ses.tolist()}"
    )


@pytest.mark.slow
def test_put_call_parity_shared_seed():
    """
    Put–call parity under shared random numbers (common random numbers).

    With shared seed, call and put are perfectly coupled through the same
    simulated S_T paths. This means most Monte Carlo noise cancels in
    (call - put), so using independent SEs would massively overestimate
    the true variance.
    """

    S0 = K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    N = 100_000
    seed = 42
    
    # Price call and put using identical random numbers (CRN)
    call = mc_price(S0, K, T, r, sigma, "call", N, seed=seed)
    put  = mc_price(S0, K, T, r, sigma, "put", N, seed=seed)
    
    lhs = call["price"] - put["price"]
    rhs = S0 - K * np.exp(-r * T)
    
    # Under shared seed:
    # C - P = e^{-rT} * (mean S_T) - K e^{-rT}
    # So the only Monte Carlo noise is from estimating E[S_T]
    
    # Estimate standard error of S_T mean directly
    # (NOT by combining call/put SEs, since those noises cancel)
    st_se = call["std_error"] * np.exp(r * T)
    
    tol = 5 * st_se
    
    assert abs(lhs - rhs) < tol, (
        f"put-call parity mismatch under shared seed:\n"
        f"call-put = {lhs:.6f}, theory = {rhs:.6f}, "
        f"tolerance = {tol:.6f}"
    )




@pytest.mark.parametrize(
    "overrides, exc",
    [
        (dict(S0=-1), ValueError),
        (dict(K=0), ValueError),
        (dict(T=-0.1), ValueError),
        (dict(sigma=-0.01), ValueError),
        (dict(n_paths=10_000.5), TypeError),
        (dict(n_paths=1), ValueError),
        (dict(option_type="straddle"), ValueError),
    ],
)
def test_input_validation(overrides, exc):
    base = dict(
        S0=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type="call",
        n_paths=10_000,
        seed=0,
    )
    base.update(overrides)

    with pytest.raises(exc):
        mc_price(**base)


def test_zero_volatility_collapses_to_intrinsic():
    S0, K, T, r = 100.0, 90.0, 1.0, 0.05
    expected = max(S0 - K * np.exp(-r * T), 0.0)

    res = mc_price(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=0.0,
        option_type="call",
        n_paths=10_000,
        seed=0,
    )

    assert abs(res["price"] - expected) < 1e-12, (
        f"price={res['price']:.12f}, expected={expected:.12f}"
    )
    assert res["std_error"] == 0.0
    assert res["ci_95"] == (expected, expected)


def test_T_zero_returns_intrinsic():
    res = mc_price(100, 90, 0.0, 0.05, 0.2, "call", 1000, seed=0)

    assert res["price"] == 10.0
    assert res["std_error"] == 0.0
    assert res["ci_95"] == (10.0, 10.0)


def test_deep_otm_call_ci_includes_zero():
    res = mc_price(50, 200, 1.0, 0.05, 0.2, "call", 100_000, seed=0)
    lo, hi = res["ci_95"]

    assert lo <= hi, f"invalid CI ordering: {res['ci_95']}"
    assert lo <= 0.0 <= hi, f"expected CI to include 0.0, got {res['ci_95']}"


def test_reproducibility():
    args = dict(
        S0=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type="call",
        n_paths=10_000,
        seed=42,
    )
    a = mc_price(**args)
    b = mc_price(**args)
    assert a == b, f"results differ for same seed: {a} vs {b}"