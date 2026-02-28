use pyo3::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::types::OptionChar;

/// Standard normal distribution (reused across the crate)
fn std_normal() -> Normal {
    Normal::new(0.0, 1.0).unwrap()
}

/// norm.cdf(x)
pub fn norm_cdf(x: f64) -> f64 {
    std_normal().cdf(x)
}

/// norm.pdf(x)
pub fn norm_pdf(x: f64) -> f64 {
    use statrs::distribution::Continuous;
    std_normal().pdf(x)
}

/// Vanilla European option pricing (Black-Scholes-Merton).
///
/// Equivalent to Python `_bsm_euro`.
pub fn bsm_euro_inner(option: OptionChar, tau: f64, vol: f64, k: f64, s: f64, r: f64) -> f64 {
    if vol == 0.0 {
        return match option {
            OptionChar::Call => (s - k).max(0.0),
            OptionChar::Put => (k - s).max(0.0),
        };
    }
    let sqrt_tau = tau.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * vol * vol) * tau) / (vol * sqrt_tau);
    let d2 = d1 - vol * sqrt_tau;
    let discount = (-r * tau).exp();
    match option {
        OptionChar::Call => {
            discount * (norm_cdf(d1) * s - norm_cdf(d2) * k)
        }
        OptionChar::Put => {
            discount * (norm_cdf(-d2) * k - norm_cdf(-d1) * s)
        }
    }
}

/// Master pricing dispatcher. Equivalent to Python `_compute_value`.
///
/// For vanilla options (no barrier), calls BSM.
/// For barrier options, delegates to barrier_amer or barrier_euro.
pub fn compute_value_inner(
    char: OptionChar,
    tau: f64,
    vol: f64,
    k: f64,
    s: f64,
    r: f64,
    _payoff: &str,
    ki: Option<f64>,
    ko: Option<f64>,
    barrier: Option<&str>,
    direction: Option<&str>,
    product: Option<&str>,
    bvol: Option<f64>,
    bvol2: Option<f64>,
    dbarrier: Option<f64>,
) -> f64 {
    // expiry case
    if tau <= 0.0 || (tau.abs() < 1e-15) {
        return match char {
            OptionChar::Call => (s - k).max(0.0),
            OptionChar::Put => (k - s).max(0.0),
        };
    }

    match barrier {
        None => {
            // vanilla option — amer and euro are treated identically
            bsm_euro_inner(char, tau, vol, k, s, r)
        }
        Some("amer") => {
            let d = direction.expect("direction required for barrier option");
            crate::barrier_amer::barrier_amer_inner(
                char, tau, vol, k, s, r, _payoff,
                crate::types::Direction::from_str(d),
                ki, ko, 0.0,
            )
        }
        Some("euro") => {
            let d = direction.expect("direction required for barrier option");
            let pdt = product.expect("product required for euro barrier");
            crate::barrier_euro::barrier_euro_inner(
                char, tau, vol, k, s, r, _payoff, d, ki, ko, pdt,
                0.0, bvol, bvol2, dbarrier,
            )
        }
        Some(other) => panic!("Unknown barrier type: '{}'", other),
    }
}

// ── Python-exposed functions ──────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "bsm_euro")]
pub fn bsm_euro(option: &str, tau: f64, vol: f64, k: f64, s: f64, r: f64) -> PyResult<f64> {
    Ok(bsm_euro_inner(OptionChar::from_str(option), tau, vol, k, s, r))
}

#[pyfunction]
#[pyo3(name = "compute_value")]
#[pyo3(signature = (char, tau, vol, k, s, r, payoff, ki=None, ko=None, barrier=None, d=None, product=None, bvol=None, bvol2=None, dbarrier=None))]
pub fn compute_value(
    char: &str,
    tau: f64,
    vol: f64,
    k: f64,
    s: f64,
    r: f64,
    payoff: &str,
    ki: Option<f64>,
    ko: Option<f64>,
    barrier: Option<&str>,
    d: Option<&str>,
    product: Option<&str>,
    bvol: Option<f64>,
    bvol2: Option<f64>,
    dbarrier: Option<f64>,
) -> PyResult<f64> {
    Ok(compute_value_inner(
        OptionChar::from_str(char),
        tau, vol, k, s, r, payoff, ki, ko, barrier, d, product, bvol, bvol2, dbarrier,
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
    fn test_vanilla_pricing() {
        let vol = 0.22;
        let s = 387.750;
        let r = 0.0;
        let strikes = [350.0, 360.0, 375.0, 387.75, 388.0, 389.0, 400.0, 420.0, 440.0,
                       440.0, 420.0, 400.0, 387.75, 388.0, 389.0, 350.0, 360.0, 370.0];
        let chars = [OptionChar::Call; 9].iter().copied()
            .chain([OptionChar::Put; 9].iter().copied())
            .collect::<Vec<_>>();
        let actuals = [
            49.005645520224000, 42.405054015026500, 33.620522291462900,
            27.213528169831200, 27.097482827734100, 26.636930698204900,
            21.947072146899300, 15.077435162573200, 10.066343062463700,
            62.316343062463700, 47.327435162573200, 34.197072146899300,
            27.2135281698312, 27.3474828277341, 27.8869306982049,
            11.2556455202240, 14.6550540150265, 18.6483083574217,
        ];

        for i in 0..strikes.len() {
            let val = bsm_euro_inner(chars[i], tau(), vol, strikes[i], s, r);
            assert!(
                (val - actuals[i]).abs() < 1e-8,
                "Strike {}: got {}, expected {}", strikes[i], val, actuals[i]
            );
        }
    }
}
