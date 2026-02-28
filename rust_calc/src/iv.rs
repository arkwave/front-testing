use pyo3::prelude::*;
use std::f64::consts::PI;

use crate::bsm::{bsm_euro_inner, norm_pdf};
use crate::multipliers::get_multipliers;
use crate::types::OptionChar;

/// Newton-Raphson IV solver for vanilla European options.
///
/// Equivalent to Python `newton_raphson`.
pub fn newton_raphson_inner(
    option: OptionChar,
    s: f64,
    k: f64,
    c: f64,
    tau: f64,
    r: f64,
    num_iter: usize,
) -> f64 {
    let precision = 1e-3;
    let mut guess = (2.0 * PI / tau).sqrt() * (c / s);

    for _ in 0..num_iter {
        if guess <= 0.0 {
            guess = 1e-8;
        }
        let sqrt_tau = tau.sqrt();
        let d1 = ((s / k).ln() + (r + 0.5 * guess * guess) * tau) / (guess * sqrt_tau);
        let option_price = bsm_euro_inner(option, tau, guess, k, s, r);
        let vega = s * norm_pdf(d1) * sqrt_tau;
        let diff = option_price - c;
        if diff.abs() < precision {
            return guess;
        }
        if vega.abs() < 1e-15 {
            break;
        }
        guess -= diff / vega;
    }

    if guess.is_nan() { 0.0 } else { guess }
}

/// Compute IV wrapper. Equivalent to Python `_compute_iv`.
pub fn compute_iv_inner(
    optiontype: OptionChar,
    s: f64,
    k: f64,
    c: f64,
    tau: f64,
    r: f64,
    _flag: &str,
) -> f64 {
    // Both 'euro' and 'amer' use Newton-Raphson in the Python code
    newton_raphson_inner(optiontype, s, k, c, tau, r, 100)
}

/// Compute strike from delta. Equivalent to Python `compute_strike_from_delta`.
///
/// This version accepts raw parameters instead of an Option object.
pub fn compute_strike_from_delta_inner(
    delta: f64,
    vol: f64,
    s: f64,
    tau: f64,
    char: OptionChar,
    product: &str,
) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let n = Normal::new(0.0, 1.0).unwrap();

    let delta = if delta == 0.0 { 1e-5 } else if delta == 1.0 { 0.99 } else { delta };

    let d = match char {
        OptionChar::Call => n.inverse_cdf(delta),
        OptionChar::Put => -n.inverse_cdf(delta),
    };

    let strike = s / ((vol * tau.sqrt() * d) - (vol * vol * tau) / 2.0).exp();

    let mult = get_multipliers(product);
    let ticksize = mult.options_tick;
    ((strike / ticksize).round() * ticksize * 100.0).round() / 100.0
}

// ── Python-exposed functions ──────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "newton_raphson")]
#[pyo3(signature = (option, s, k, c, tau, r, num_iter=100))]
pub fn py_newton_raphson(
    option: &str,
    s: f64,
    k: f64,
    c: f64,
    tau: f64,
    r: f64,
    num_iter: usize,
) -> PyResult<f64> {
    Ok(newton_raphson_inner(
        OptionChar::from_str(option), s, k, c, tau, r, num_iter,
    ))
}

#[pyfunction]
#[pyo3(name = "compute_iv")]
pub fn py_compute_iv(
    optiontype: &str,
    s: f64,
    k: f64,
    c: f64,
    tau: f64,
    r: f64,
    flag: &str,
) -> PyResult<f64> {
    Ok(compute_iv_inner(
        OptionChar::from_str(optiontype), s, k, c, tau, r, flag,
    ))
}

#[pyfunction]
#[pyo3(name = "compute_strike_from_delta")]
pub fn py_compute_strike_from_delta(
    delta: f64,
    vol: f64,
    s: f64,
    tau: f64,
    char: &str,
    product: &str,
) -> PyResult<f64> {
    Ok(compute_strike_from_delta_inner(
        delta, vol, s, tau, OptionChar::from_str(char), product,
    ))
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tau() -> f64 {
        233.0 / 365.0 + 1.0 / 365.0
    }

    #[test]
    fn test_newton_raphson_roundtrip() {
        let s = 387.750;
        let r = 0.0;
        let vol = 0.22;
        let k = 387.75;

        // Price a call, then recover vol
        let price = bsm_euro_inner(OptionChar::Call, tau(), vol, k, s, r);
        let iv = newton_raphson_inner(OptionChar::Call, s, k, price, tau(), r, 100);
        assert!((iv - vol).abs() < 1e-3, "IV roundtrip: got {}, expected {}", iv, vol);

        // Price a put, then recover vol
        let price = bsm_euro_inner(OptionChar::Put, tau(), vol, k, s, r);
        let iv = newton_raphson_inner(OptionChar::Put, s, k, price, tau(), r, 100);
        assert!((iv - vol).abs() < 1e-3, "IV roundtrip put: got {}, expected {}", iv, vol);
    }
}
