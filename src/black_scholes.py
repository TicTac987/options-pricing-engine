import numpy as np
from scipy.stats import norm
from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt

def bs_price(S0, K, T, r, sigma, option_type: Literal["call", "put"]):
    """
    Price a European option under the Black-Scholes-Merton model.

    Computes the arbitrage-free price of a European call or put option using
    the closed-form Black-Scholes formula under standard assumptions:
    constant volatility, lognormal dynamics, continuous compounding, and no
    transaction costs.

    Parameters
    ----------
    S0 : float or np.ndarray
        Current underlying asset price.
    K : float or np.ndarray
        Strike price.
    T : float or np.ndarray
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    option_type : {"call", "put"}
        Type of option to price.

    Returns
    -------
    float or np.ndarray
        Black-Scholes price of the option. Output type matches input broadcasting.
    """
    
    S0 = np.asarray(S0, dtype=float)
    K  = np.asarray(K, dtype=float)
    T = float(T)
    
    if np.any(S0 <= 0) or np.any(K <= 0):
        raise ValueError("S0 and K must be positive")
        
    if T < 0:   
        raise ValueError("T must be non-negative")
        
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
        
    # Near expiry
    if T < 1e-9:
        if option_type == 'call':
            return np.maximum(S0 - K, 0.0)
        
        elif option_type == 'put':
            return np.maximum(K - S0, 0.0)
        
        else:
            raise ValueError("Invalid option_type")
    
    # Zero volatility
    sqrtT = np.sqrt(T)
    disc  = np.exp(-r*T)
    
    if sigma <= 1e-10:
        forward = S0 * np.exp(r*T)
        
        if option_type == "call":
            return disc * np.maximum(forward - K, 0.0)
        
        else:
            return disc * np.maximum(K - forward, 0.0)
        
    d1 = (np.log(S0) - np.log(K) + (r + 0.5*sigma**2)*T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    
    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    
    elif option_type == 'put':
        return K * disc * norm.cdf(-d2) - S0* norm.cdf(-d1)
        
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        
        
def bs_put_call_parity_check(S0, K, T, r, sigma):
    """
    Validate Black-Scholes put-call parity and quantify numerical error.

    Checks the fundamental no-arbitrage identity:
        C - P = S0 - K * exp(-rT)

    where C and P are European call and put prices under the Black-Scholes
    model. The function computes both sides independently and returns both
    absolute and scaled error metrics.

    Parameters
    ----------
    S0 : float or np.ndarray
        Current underlying asset price.
    K : float or np.ndarray
        Strike price.
    T : float or np.ndarray
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.

    Returns
    -------
    residual : np.ndarray
        Absolute pricing error:
            (C - P) - (S0 - K * exp(-rT))
        Should be approximately zero up to numerical precision.

    rel_error : np.ndarray
        Relative error scaled by max(1, |S0 - K * exp(-rT)|) to ensure
        numerical stability across different price magnitudes.
    """
    
    S0 = np.asarray(S0, dtype=float)
    K  = np.asarray(K, dtype=float)
    T  = np.asarray(T, dtype=float)
    
    # Compute prices
    C = bs_price(S0, K, T, r, sigma, option_type='call')
    P = bs_price(S0, K, T, r, sigma, option_type='put')
    
    # RHS of parity
    disc = np.exp(-r * T)
    rhs  = S0 - K * disc
    
    residual = (C - P) - rhs
    
    # Relative error
    scale     = np.maximum(1.0, np.abs(rhs)) 
    rel_error = residual / scale
    
    return residual, rel_error


def assert_put_call_parity(S0, K, T, r, sigma, tol=1e-10):
    """
    Assert Black-Scholes put-call parity holds within numerical tolerance.
    
    Verifies the identity:
        C - P = S0 - K * exp(-rT)
    
    where:
        C = European call price
        P = European put price
    
    This function computes both call and put prices using the Black-Scholes
    model and checks that the parity residual is close to zero up to a
    specified tolerance.
    
    Parameters
    ----------
    S0 : float or np.ndarray
        Current underlying asset price.
    K : float or np.ndarray
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    tol : float, optional
        Maximum allowed relative error tolerance (default is 1e-10).
    
    Raises
    ------
    AssertionError
        If the absolute relative error exceeds the specified tolerance.
    
    Returns
    -------
    None
    """
    residual, rel_error = bs_put_call_parity_check(S0, K, T, r, sigma)
    
    if not np.all(np.abs(rel_error) < tol):
        raise AssertionError(
            f"Put-call parity violated. Max rel error: {np.max(np.abs(rel_error)):.2e}"
        )
        

def hull_verification_table(bs_price_func):
    """
    Construct a Black-Scholes pricing verification table using benchmark values
    from Hull-textbook-style test cases.
    
    This function evaluates a set of standard European option scenarios and
    compares computed Black-Scholes prices against reference values from
    financial literature (e.g., Hull's *Options, Futures, and Other Derivatives*).
    
    Parameters
    ----------
    bs_price_func : callable
        Black-Scholes pricing function with signature:
        bs_price(S0, K, T, r, sigma, option_type)
    
    Returns
    -------
    pandas.DataFrame
        Table containing:
        - Input parameters (S0, K, T, sigma, option type)
        - Model price
        - Reference price
        - Absolute pricing error
    
    Columns
    -------
    S0 : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    sigma : float
        Volatility
    type : str
        Option type ('call' or 'put')
    model : float
        Black-Scholes computed price
    reference : float
        Benchmark price from literature
    error : float
        Model - reference pricing error
    abs_error : float
        Absolute pricing error
    """
    
    tests = [
    # S0,   K,    T,     r,    sigma,  type,   reference_price
    (100,  100,  1.0,   0.05,  0.20,  "call", 10.450584),
    (100,  100,  1.0,   0.05,  0.20,  "put",   5.573526),
    (100,  100,  1.0,   0.05,  0.30,  "call", 14.231255),
    (105,  100,  1.0,   0.05,  0.20,  "call", 13.857906),
    (95,   100,  1.0,   0.05,  0.20,  "call",  7.510872),
    (100,  100,  0.25,  0.05,  0.20,  "call",  4.614997),
    (100,  100,  1.0,   0.00,  0.20,  "call",  7.965567),
    (100,  100,  1.0,   0.05,  0.50,  "call", 21.792604),
    (100,   90,  1.0,   0.05,  0.20,  "call", 16.699448),
    (100,  110,  1.0,   0.05,  0.20,  "put",  10.675325),
    (100,  100,  0.083333, 0.05, 0.20, "call", 2.512062),   # ~1 month
    (100,  100,  1.0,   0.05,  0.20,  "put",   5.573526),   # repeat for put-call parity check
    ]    
    
    rows = []
    
    for S0, K, T, r, sigma, opt_type, ref in tests:
        model = bs_price_func(S0, K, T, r, sigma, opt_type)
        error = model - ref
        
        rows.append([
            S0, K, T, r, sigma, opt_type, model, ref, error
            ])
        
    df = pd.DataFrame(
        rows,
        columns=["S0", "K", "T", "r", "sigma", "type", "model", "reference", "error"]
    )       
    
    df["abs_error"] = df["error"].abs()
    
    return df


def plot_call_vs_spot(K=100.0, T=1.0, r=0.05, sigma=0.20,
                      S0_range=(60, 140), n=200):
    """
    Plot Black-Scholes European call price and intrinsic value vs. spot price.

    Evaluates the BS call price and intrinsic payoff max(S0 - K, 0) across a
    range of spot prices, illustrating how time value smooths the hockey-stick
    payoff shape.

    Parameters
    ----------
    K       : float  Strike price.                     Default: 100
    T       : float  Time to maturity (years).         Default: 1.0
    r       : float  Risk-free rate.                   Default: 0.05
    sigma   : float  Volatility.                       Default: 0.20
    S0_range: tuple  (S0_min, S0_max) for x-axis.     Default: (60, 140)
    n       : int    Number of spot price grid points. Default: 200

    Returns
    -------
    None
        Displays a matplotlib figure.
    """
    
    S0_vals = np.linspace(*S0_range, n)    
    prices    = bs_price(S0_vals, K, T, r, sigma, "call")
    intrinsic = np.maximum(S0_vals - K, 0)
        
    d1    = ( (np.log(S0_vals) - np.log(K) + (r + 0.5*sigma**2)*T) 
             / (sigma * np.sqrt(T)) )
    delta = norm.cdf(d1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Price vs Spot ---
    ax1.plot(S0_vals, prices,    label="BS call price")
    ax1.plot(S0_vals, intrinsic, label="Intrinsic value", linestyle="--", color="grey")
    ax1.set_xlabel("Spot price $S_0$")
    ax1.set_ylabel("Price")
    ax1.set_title("BS Call Price vs Spot")
    ax1.legend()
    ax1.grid(True)
    
    # --- Right: Delta vs Spot ---
    ax2.plot(S0_vals, delta, color="tab:orange")
    ax2.set_xlabel("Spot price $S_0$")
    ax2.set_ylabel(r"$\Delta = N(d_1)$")
    ax2.set_title("Call Delta vs Spot")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    

# if __name__ == "__main__":
#     plot_call_vs_spot()
#     pd.set_option("display.float_format", "{:.6f}".format)
#     df = hull_verification_table(bs_price)
#     print(df.to_string(index=False))