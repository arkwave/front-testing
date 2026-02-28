"""
Drop-in replacement for calc.py that uses Rust-backed implementations
where available. Falls back to pure Python if rust_calc is not installed.

Usage: change `from scripts.calc import ...` to `from scripts.calc_rs import ...`
Original calc.py remains untouched.
"""

try:
    import rust_calc
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# DataFrame-dependent functions stay in Python (re-exported from calc.py)
from scripts.calc import get_barrier_vol, get_vol_from_delta, get_vol_at_strike, compute_delta

# multipliers dict re-exported from calc.py
from scripts.calc import multipliers


# ── Pricing functions ──────────────────────────────────────────────────

def _bsm_euro(option, tau, vol, K, s, r):
    if _RUST_AVAILABLE:
        return rust_calc.bsm_euro(option, tau, vol, K, s, r)
    from scripts.calc import _bsm_euro as _py
    return _py(option, tau, vol, K, s, r)


def _compute_value(char, tau, vol, K, s, r, payoff, ki=None, ko=None,
                   barrier=None, d=None, product=None, bvol=None, bvol2=None,
                   dbarrier=None):
    if _RUST_AVAILABLE:
        return rust_calc.compute_value(char, tau, vol, K, s, r, payoff,
                                       ki=ki, ko=ko, barrier=barrier, d=d,
                                       product=product, bvol=bvol, bvol2=bvol2,
                                       dbarrier=dbarrier)
    from scripts.calc import _compute_value as _py
    return _py(char, tau, vol, K, s, r, payoff, ki=ki, ko=ko,
               barrier=barrier, d=d, product=product, bvol=bvol,
               bvol2=bvol2, dbarrier=dbarrier)


def _barrier_amer(char, tau, vol, k, s, r, payoff, direction, ki, ko, rebate=0):
    if _RUST_AVAILABLE:
        return rust_calc.barrier_amer(char, tau, vol, k, s, r, payoff,
                                      direction, ki=ki, ko=ko, rebate=rebate)
    from scripts.calc import _barrier_amer as _py
    return _py(char, tau, vol, k, s, r, payoff, direction, ki, ko, rebate=rebate)


def _barrier_euro(char, tau, vol, k, s, r, payoff, direction,
                  ki, ko, product, rebate=0, bvol=None, bvol2=None,
                  dbarrier=None):
    if _RUST_AVAILABLE:
        return rust_calc.barrier_euro(char, tau, vol, k, s, r, payoff,
                                      direction, ki=ki, ko=ko, product=product,
                                      rebate=rebate, bvol=bvol, bvol2=bvol2,
                                      dbarrier=dbarrier)
    from scripts.calc import _barrier_euro as _py
    return _py(char, tau, vol, k, s, r, payoff, direction, ki, ko, product,
               rebate=rebate, bvol=bvol, bvol2=bvol2, dbarrier=dbarrier)


def digital_option(char, tau, vol, dbarvol, k, dbar, s, r, payoff, product):
    if _RUST_AVAILABLE:
        return rust_calc.digital_option(char, tau, vol, dbarvol, k, dbar, s, r, payoff, product)
    from scripts.calc import digital_option as _py
    return _py(char, tau, vol, dbarvol, k, dbar, s, r, payoff, product)


def call_put_spread(s, k1, k2, r, vol1, vol2, tau, optiontype, payoff, b=0):
    if _RUST_AVAILABLE:
        return rust_calc.call_put_spread(s, k1, k2, r, vol1, vol2, tau, optiontype, payoff, b=b)
    from scripts.calc import call_put_spread as _py
    return _py(s, k1, k2, r, vol1, vol2, tau, optiontype, payoff, b=b)


# ── Greeks functions ───────────────────────────────────────────────────

def _compute_greeks(char, K, tau, vol, s, r, product, payoff, lots,
                    ki=None, ko=None, barrier=None, direction=None,
                    order=None, bvol=None, bvol2=None, dbarrier=None):
    if _RUST_AVAILABLE:
        return rust_calc.compute_greeks(char, K, tau, vol, s, r, product, payoff, lots,
                                        ki=ki, ko=ko, barrier=barrier, direction=direction,
                                        order=order, bvol=bvol, bvol2=bvol2, dbarrier=dbarrier)
    from scripts.calc import _compute_greeks as _py
    return _py(char, K, tau, vol, s, r, product, payoff, lots,
               ki=ki, ko=ko, barrier=barrier, direction=direction,
               order=order, bvol=bvol, bvol2=bvol2, dbarrier=dbarrier)


def _euro_vanilla_greeks(char, K, tau, vol, s, r, product, lots):
    if _RUST_AVAILABLE:
        return rust_calc.euro_vanilla_greeks(char, K, tau, vol, s, r, product, lots)
    from scripts.calc import _euro_vanilla_greeks as _py
    return _py(char, K, tau, vol, s, r, product, lots)


def _euro_barrier_amer_greeks(char, tau, vol, k, s, r, payoff, direction,
                               product, ki, ko, lots, rebate=0):
    if _RUST_AVAILABLE:
        return rust_calc.euro_barrier_amer_greeks(char, tau, vol, k, s, r, payoff,
                                                   direction, product, ki=ki, ko=ko,
                                                   lots=lots, rebate=rebate)
    from scripts.calc import _euro_barrier_amer_greeks as _py
    return _py(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, lots, rebate=rebate)


def _euro_barrier_euro_greeks(char, tau, vol, k, s, r, payoff, direction,
                               product, ki, ko, lots, order=None, rebate=0,
                               bvol=None, bvol2=None, dbarrier=None):
    if _RUST_AVAILABLE:
        return rust_calc.euro_barrier_euro_greeks(char, tau, vol, k, s, r, payoff,
                                                   direction, product, ki=ki, ko=ko,
                                                   lots=lots, order=order, rebate=rebate,
                                                   bvol=bvol, bvol2=bvol2, dbarrier=dbarrier)
    from scripts.calc import _euro_barrier_euro_greeks as _py
    return _py(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, lots,
               order=order, rebate=rebate, bvol=bvol, bvol2=bvol2, dbarrier=dbarrier)


def digital_greeks(char, k, dbar, tau, vol, vol2, s, r, product, payoff, lots):
    if _RUST_AVAILABLE:
        return rust_calc.digital_greeks(char, k, dbar, tau, vol, vol2, s, r, product, payoff, lots)
    from scripts.calc import digital_greeks as _py
    return _py(char, k, dbar, tau, vol, vol2, s, r, product, payoff, lots)


def call_put_spread_greeks(s, k1, k2, r, vol1, vol2, tau, optiontype, product, lots, payoff, b=0):
    if _RUST_AVAILABLE:
        return rust_calc.call_put_spread_greeks(s, k1, k2, r, vol1, vol2, tau,
                                                optiontype, product, lots, payoff, b=b)
    from scripts.calc import call_put_spread_greeks as _py
    return _py(s, k1, k2, r, vol1, vol2, tau, optiontype, product, lots, payoff, b=b)


def greeks_scaled(delta1, gamma1, theta1, vega1, product, lots):
    if _RUST_AVAILABLE:
        return rust_calc.greeks_scaled(delta1, gamma1, theta1, vega1, product, lots)
    from scripts.calc import greeks_scaled as _py
    return _py(delta1, gamma1, theta1, vega1, product, lots)


# ── IV functions ───────────────────────────────────────────────────────

def newton_raphson(option, s, k, c, tau, r, num_iter=100):
    if _RUST_AVAILABLE:
        return rust_calc.newton_raphson(option, s, k, c, tau, r, num_iter=num_iter)
    from scripts.calc import newton_raphson as _py
    return _py(option, s, k, c, tau, r, num_iter=num_iter)


def _compute_iv(optiontype, s, k, c, tau, r, flag):
    if _RUST_AVAILABLE:
        return rust_calc.compute_iv(optiontype, s, k, c, tau, r, flag)
    from scripts.calc import _compute_iv as _py
    return _py(optiontype, s, k, c, tau, r, flag)


def compute_strike_from_delta(option=None, delta1=None, vol=None, s=None,
                               tau=None, char=None, pdt=None):
    if _RUST_AVAILABLE and option is None:
        # Rust version requires raw params, not an Option object
        return rust_calc.compute_strike_from_delta(delta1, vol, s, tau, char, pdt)
    from scripts.calc import compute_strike_from_delta as _py
    return _py(option, delta1=delta1, vol=vol, s=s, tau=tau, char=char, pdt=pdt)
