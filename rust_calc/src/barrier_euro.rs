use pyo3::prelude::*;

use crate::bsm::{bsm_euro_inner, compute_value_inner};
use crate::multipliers::get_multipliers;
use crate::types::OptionChar;

/// Digital option pricing — used in European barrier valuation.
///
/// Equivalent to Python `digital_option`.
pub fn digital_option_inner(
    char: OptionChar,
    tau: f64,
    vol: f64,
    dbarvol: f64,
    k: f64,
    dbar: f64,
    s: f64,
    r: f64,
    payoff: &str,
    product: &str,
) -> f64 {
    let ticksize = get_multipliers(product).futures_tick;
    if tau <= 0.0 {
        return match char {
            OptionChar::Call if s >= k => ticksize,
            OptionChar::Put if s <= k => ticksize,
            _ => 0.0,
        };
    }
    let c1 = compute_value_inner(char, tau, dbarvol, dbar, s, r, payoff, None, None, None, None, None, None, None, None);
    let c2 = compute_value_inner(char, tau, vol, k, s, r, payoff, None, None, None, None, None, None, None, None);
    c1 - c2
}

/// Call/put spread pricing.
///
/// Equivalent to Python `call_put_spread`.
pub fn call_put_spread_inner(
    s: f64,
    k1: f64,
    k2: f64,
    r: f64,
    vol1: f64,
    vol2: f64,
    tau: f64,
    optiontype: &str,
    payoff: &str,
) -> f64 {
    match optiontype {
        "callspread" => {
            let p1 = bsm_euro_inner(OptionChar::Call, tau, vol1, k1, s, r);
            let p2 = bsm_euro_inner(OptionChar::Call, tau, vol2, k2, s, r);
            p2 - p1
        }
        "putspread" => {
            let p1 = bsm_euro_inner(OptionChar::Put, tau, vol1, k1, s, r);
            let p2 = bsm_euro_inner(OptionChar::Put, tau, vol2, k2, s, r);
            p1 - p2
        }
        _ => panic!("Unknown option type: '{}'", optiontype),
    }
}

/// European barrier option pricing via call-spread + digital replication.
///
/// Equivalent to Python `_barrier_euro`.
pub fn barrier_euro_inner(
    char: OptionChar,
    tau: f64,
    vol: f64,
    k: f64,
    s: f64,
    r: f64,
    payoff: &str,
    direction: &str,
    ki: Option<f64>,
    ko: Option<f64>,
    product: &str,
    _rebate: f64,
    bvol: Option<f64>,
    bvol2: Option<f64>,
    dbarrier: Option<f64>,
) -> f64 {
    let barlevel = ki.or(ko).expect("Either ki or ko must be provided");
    let mult = get_multipliers(product);
    let ticksize = mult.futures_tick;

    let dbarrier = dbarrier.unwrap_or_else(|| {
        if direction == "up" {
            barlevel - ticksize
        } else {
            barlevel + ticksize
        }
    });

    let dpo = (k - barlevel).abs() / ticksize;
    let bvol_val = bvol.unwrap_or(vol);
    let bvol2_val = bvol2.unwrap_or(vol);

    if ko.is_some() {
        let c1 = compute_value_inner(char, tau, vol, k, s, r, payoff, None, None, None, None, None, None, None, None);
        let c2 = compute_value_inner(char, tau, bvol_val, barlevel, s, r, payoff, None, None, None, None, None, None, None, None);
        let c3 = digital_option_inner(char, tau, bvol_val, bvol2_val, barlevel, dbarrier, s, r, payoff, product) * dpo;
        c1 - c2 - c3
    } else {
        // ki
        let c1 = compute_value_inner(char, tau, bvol_val, barlevel, s, r, payoff, None, None, None, None, None, None, None, None);
        let c2 = dpo * digital_option_inner(char, tau, bvol_val, bvol2_val, barlevel, dbarrier, s, r, payoff, product);
        c1 + c2
    }
}

// ── Python-exposed functions ──────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "barrier_euro")]
#[pyo3(signature = (char, tau, vol, k, s, r, payoff, direction, ki=None, ko=None, product="C", rebate=0.0, bvol=None, bvol2=None, dbarrier=None))]
pub fn barrier_euro(
    char: &str,
    tau: f64,
    vol: f64,
    k: f64,
    s: f64,
    r: f64,
    payoff: &str,
    direction: &str,
    ki: Option<f64>,
    ko: Option<f64>,
    product: &str,
    rebate: f64,
    bvol: Option<f64>,
    bvol2: Option<f64>,
    dbarrier: Option<f64>,
) -> PyResult<f64> {
    Ok(barrier_euro_inner(
        OptionChar::from_str(char),
        tau, vol, k, s, r, payoff, direction, ki, ko, product, rebate, bvol, bvol2, dbarrier,
    ))
}

#[pyfunction]
#[pyo3(name = "digital_option")]
pub fn py_digital_option(
    char: &str,
    tau: f64,
    vol: f64,
    dbarvol: f64,
    k: f64,
    dbar: f64,
    s: f64,
    r: f64,
    payoff: &str,
    product: &str,
) -> PyResult<f64> {
    Ok(digital_option_inner(
        OptionChar::from_str(char),
        tau, vol, dbarvol, k, dbar, s, r, payoff, product,
    ))
}

#[pyfunction]
#[pyo3(name = "call_put_spread")]
#[pyo3(signature = (s, k1, k2, r, vol1, vol2, tau, optiontype, payoff, b=0.0))]
pub fn py_call_put_spread(
    s: f64,
    k1: f64,
    k2: f64,
    r: f64,
    vol1: f64,
    vol2: f64,
    tau: f64,
    optiontype: &str,
    payoff: &str,
    b: f64,
) -> PyResult<f64> {
    Ok(call_put_spread_inner(s, k1, k2, r, vol1, vol2, tau, optiontype, payoff))
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tau() -> f64 {
        233.0 / 365.0 + 1.0 / 365.0
    }

    #[test]
    fn test_euro_barrier_pricing() {
        let s = 387.750;
        let vol = 0.22;
        let chars = [OptionChar::Call, OptionChar::Call, OptionChar::Put, OptionChar::Put];
        let directions = ["up", "up", "down", "down"];
        let kis: [Option<f64>; 4] = [None, Some(390.0), None, Some(370.0)];
        let kos: [Option<f64>; 4] = [Some(400.0), None, Some(350.0), None];
        let strikes = [350.0, 360.0, 390.0, 400.0];
        let actuals = [
            7.242409174727800, 39.760087685635000,
            4.713297229131670, 31.552517740371500,
        ];

        for i in 0..4 {
            let val = barrier_euro_inner(
                chars[i], tau(), vol, strikes[i], s, 0.0,
                "amer", directions[i], kis[i], kos[i],
                "C", 0.0, Some(vol), Some(vol), None,
            );
            assert!(
                (val - actuals[i]).abs() < 1e-6,
                "Case {}: got {}, expected {}", i, val, actuals[i]
            );
        }
    }
}
