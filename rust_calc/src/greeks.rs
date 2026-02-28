use pyo3::prelude::*;

use crate::barrier_amer::barrier_amer_inner;
use crate::bsm::{norm_cdf, norm_pdf};
use crate::multipliers::get_multipliers;
use crate::types::{Direction, OptionChar};

/// Scale raw BSM greeks into reportable units (PnP convention).
///
/// Equivalent to Python `greeks_scaled`.
pub fn greeks_scaled_inner(
    delta1: f64,
    gamma1: f64,
    theta1: f64,
    vega1: f64,
    product: &str,
    lots: f64,
) -> (f64, f64, f64, f64) {
    let mult = get_multipliers(product);
    let lm = mult.lot_mult;
    let dm = mult.dollar_mult;
    let pnl_mult = mult.pnl_mult;

    let delta = delta1 * lots;
    let gamma = (gamma1 * lots * lm) / dm;
    let vega = (vega1 * lots * pnl_mult) / 100.0;
    let theta = (theta1 * lots * pnl_mult) / 365.0;

    (delta, gamma, theta, vega)
}

/// Analytical greeks for vanilla European options.
///
/// Equivalent to Python `_euro_vanilla_greeks`.
pub fn euro_vanilla_greeks_inner(
    char: OptionChar,
    k: f64,
    tau: f64,
    vol: f64,
    s: f64,
    r: f64,
    product: &str,
    lots: f64,
) -> (f64, f64, f64, f64) {
    if vol == 0.0 {
        let (gamma, theta, vega) = (0.0, 0.0, 0.0);
        let delta = match char {
            OptionChar::Call => if k >= s { 1.0 } else { 0.0 },
            OptionChar::Put => if k >= s { -1.0 } else { 0.0 },
        };
        return (delta, theta, gamma, vega);
    }

    let sqrt_tau = tau.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * vol * vol) * tau) / (vol * sqrt_tau);
    let pdf_d1 = norm_pdf(d1);

    let gamma1 = pdf_d1 / (s * vol * sqrt_tau);
    let vega1 = s * (r * tau).exp() * pdf_d1 * sqrt_tau;

    let (delta1, theta1) = match char {
        OptionChar::Call => {
            let delta1 = norm_cdf(d1);
            let theta1 = (-s * pdf_d1 * vol) / (2.0 * sqrt_tau);
            (delta1, theta1)
        }
        OptionChar::Put => {
            let delta1 = norm_cdf(d1) - 1.0;
            let theta1 = (-s * pdf_d1 * vol) / (2.0 * sqrt_tau);
            (delta1, theta1)
        }
    };

    greeks_scaled_inner(delta1, gamma1, theta1, vega1, product, lots)
}

/// Numerical greeks for European options with American barriers (finite differences).
///
/// Equivalent to Python `_euro_barrier_amer_greeks`.
pub fn euro_barrier_amer_greeks_inner(
    char: OptionChar,
    tau: f64,
    vol: f64,
    k: f64,
    s: f64,
    r: f64,
    payoff: &str,
    direction: Direction,
    product: &str,
    ki: Option<f64>,
    ko: Option<f64>,
    lots: f64,
) -> (f64, f64, f64, f64) {
    let change_spot = 0.0005;
    let change_vol = 0.01;
    let change_tau = 1.0 / 365.0;

    let price = |c, t, v, k2, s2| {
        barrier_amer_inner(c, t, v, k2, s2, r, payoff, direction, ki, ko, 0.0)
    };

    let init = price(char, tau, vol, k, s);
    let del1 = price(char, tau, vol, k, s + change_spot);
    let del2 = price(char, tau, vol, k, (s - change_spot).max(0.0));

    let delta = (del1 - del2) / (2.0 * change_spot);
    let gamma = (del1 - 2.0 * init + del2) / (change_spot * change_spot);

    // vega
    let v1 = price(char, tau, vol + change_vol, k, s);
    let tvol = (vol - change_vol).max(0.0);
    let v2 = price(char, tau, tvol, k, s);
    let vega = if tau > 0.0 { (v1 - v2) / (2.0 * change_vol) } else { 0.0 };

    // theta
    let ctau = if tau - change_tau <= 0.0 { 0.0001 } else { tau - change_tau };
    let t2 = price(char, ctau, vol, k, s);
    let theta = if tau > 0.0 { (t2 - init) / change_tau } else { 0.0 };

    greeks_scaled_inner(delta, gamma, theta, vega, product, lots)
}

/// Digital option greeks.
///
/// Equivalent to Python `digital_greeks`.
pub fn digital_greeks_inner(
    char: OptionChar,
    k: f64,
    dbar: f64,
    tau: f64,
    vol: f64,
    vol2: f64,
    s: f64,
    r: f64,
    product: &str,
    payoff: &str,
    lots: f64,
) -> (f64, f64, f64, f64) {
    let (d1, g1, t1, v1) = compute_greeks_inner(
        char, dbar, tau, vol2, s, r, product, payoff, lots,
        None, None, None, None, None, None, None, None,
    );
    let (d2, g2, t2, v2) = compute_greeks_inner(
        char, k, tau, vol, s, r, product, payoff, lots,
        None, None, None, None, None, None, None, None,
    );
    (d1 - d2, g1 - g2, t1 - t2, v1 - v2)
}

/// Greeks for European options with European barriers (decomposition method).
///
/// Equivalent to Python `_euro_barrier_euro_greeks`.
pub fn euro_barrier_euro_greeks_inner(
    char: OptionChar,
    tau: f64,
    vol: f64,
    k: f64,
    s: f64,
    r: f64,
    payoff: &str,
    direction: &str,
    product: &str,
    ki: Option<f64>,
    ko: Option<f64>,
    lots: f64,
    bvol: Option<f64>,
    bvol2: Option<f64>,
    dbarrier: Option<f64>,
) -> (f64, f64, f64, f64) {
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
        let g1 = compute_greeks_inner(
            char, k, tau, vol, s, r, product, payoff, lots,
            None, None, None, None, None, None, None, None,
        );
        let g2 = compute_greeks_inner(
            char, barlevel, tau, bvol_val, s, r, product, payoff, lots,
            None, None, None, None, None, None, None, None,
        );
        let g3 = digital_greeks_inner(
            char, barlevel, dbarrier, tau, bvol_val, bvol2_val, s, r, product, payoff, lots,
        );

        let d = g1.0 - g2.0 - g3.0 * dpo;
        let g = g1.1 - g2.1 - g3.1 * dpo;
        let t = g1.2 - g2.2 - g3.2 * dpo;
        let v = g1.3 - g2.3 - g3.3 * dpo;
        (d, g, t, v)
    } else {
        // ki
        let g1 = compute_greeks_inner(
            char, barlevel, tau, bvol_val, s, r, product, payoff, lots,
            None, None, None, None, None, None, None, None,
        );
        let g2 = digital_greeks_inner(
            char, barlevel, dbarrier, tau, bvol_val, bvol2_val, s, r, product, payoff, lots,
        );

        let d = g1.0 + g2.0 * dpo;
        let g = g1.1 + g2.1 * dpo;
        let t = g1.2 + g2.2 * dpo;
        let v = g1.3 + g2.3 * dpo;
        (d, g, t, v)
    }
}

/// Call/put spread greeks.
///
/// Equivalent to Python `call_put_spread_greeks`.
pub fn call_put_spread_greeks_inner(
    s: f64,
    k1: f64,
    k2: f64,
    r: f64,
    vol1: f64,
    vol2: f64,
    tau: f64,
    optiontype: &str,
    product: &str,
    lots: f64,
    payoff: &str,
) -> (f64, f64, f64, f64) {
    match optiontype {
        "callspread" => {
            let (d1, g1, t1, v1) = compute_greeks_inner(
                OptionChar::Call, k1, tau, vol1, s, r, product, payoff, lots,
                None, None, None, None, None, None, None, None,
            );
            let (d2, g2, t2, v2) = compute_greeks_inner(
                OptionChar::Call, k2, tau, vol2, s, r, product, payoff, lots,
                None, None, None, None, None, None, None, None,
            );
            (d2 - d1, g2 - g1, t2 - t1, v2 - v1)
        }
        "putspread" => {
            let (d1, g1, t1, v1) = compute_greeks_inner(
                OptionChar::Put, k1, tau, vol1, s, r, product, payoff, lots,
                None, None, None, None, None, None, None, None,
            );
            let (d2, g2, t2, v2) = compute_greeks_inner(
                OptionChar::Put, k2, tau, vol2, s, r, product, payoff, lots,
                None, None, None, None, None, None, None, None,
            );
            (-(d2 - d1), -(g2 - g1), -(t2 - t1), -(v2 - v1))
        }
        _ => panic!("Unknown option type: '{}'", optiontype),
    }
}

/// Master greeks dispatcher.
///
/// Equivalent to Python `_compute_greeks`.
pub fn compute_greeks_inner(
    char: OptionChar,
    k: f64,
    tau: f64,
    vol: f64,
    s: f64,
    r: f64,
    product: &str,
    payoff: &str,
    lots: f64,
    ki: Option<f64>,
    ko: Option<f64>,
    barrier: Option<&str>,
    direction: Option<&str>,
    _order: Option<i32>,
    bvol: Option<f64>,
    bvol2: Option<f64>,
    dbarrier: Option<f64>,
) -> (f64, f64, f64, f64) {
    // tau == 0 case
    if tau == 0.0 {
        let delta = match char {
            OptionChar::Call => if k < s { 1.0 } else { 0.0 },
            OptionChar::Put => if k > s { -1.0 } else { 0.0 },
        };
        return (delta, 0.0, 0.0, 0.0);
    }

    match barrier {
        None => {
            // vanilla case
            euro_vanilla_greeks_inner(char, k, tau, vol, s, r, product, lots)
        }
        Some("amer") => {
            let dir = Direction::from_str(direction.expect("direction required for barrier greeks"));
            euro_barrier_amer_greeks_inner(char, tau, vol, k, s, r, payoff, dir, product, ki, ko, lots)
        }
        Some("euro") => {
            let dir_str = direction.expect("direction required for barrier greeks");
            let mult = get_multipliers(product);

            let dbarrier = dbarrier.or_else(|| {
                let barlevel = ki.or(ko).unwrap();
                let ticksize = mult.futures_tick;
                Some(if dir_str == "up" { barlevel - ticksize } else { barlevel + ticksize })
            });

            euro_barrier_euro_greeks_inner(
                char, tau, vol, k, s, r, payoff, dir_str, product,
                ki, ko, lots, bvol, bvol2, dbarrier,
            )
        }
        Some(other) => panic!("Unknown barrier type: '{}'", other),
    }
}

// ── Python-exposed functions ──────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "compute_greeks")]
#[pyo3(signature = (char, k, tau, vol, s, r, product, payoff, lots, ki=None, ko=None, barrier=None, direction=None, order=None, bvol=None, bvol2=None, dbarrier=None))]
pub fn py_compute_greeks(
    char: &str, k: f64, tau: f64, vol: f64, s: f64, r: f64,
    product: &str, payoff: &str, lots: f64,
    ki: Option<f64>, ko: Option<f64>, barrier: Option<&str>,
    direction: Option<&str>, order: Option<i32>,
    bvol: Option<f64>, bvol2: Option<f64>, dbarrier: Option<f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(compute_greeks_inner(
        OptionChar::from_str(char), k, tau, vol, s, r, product, payoff, lots,
        ki, ko, barrier, direction, order, bvol, bvol2, dbarrier,
    ))
}

#[pyfunction]
#[pyo3(name = "euro_vanilla_greeks")]
pub fn py_euro_vanilla_greeks(
    char: &str, k: f64, tau: f64, vol: f64, s: f64, r: f64,
    product: &str, lots: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(euro_vanilla_greeks_inner(
        OptionChar::from_str(char), k, tau, vol, s, r, product, lots,
    ))
}

#[pyfunction]
#[pyo3(name = "euro_barrier_amer_greeks")]
#[pyo3(signature = (char, tau, vol, k, s, r, payoff, direction, product, ki=None, ko=None, lots=1.0, rebate=0.0))]
pub fn py_euro_barrier_amer_greeks(
    char: &str, tau: f64, vol: f64, k: f64, s: f64, r: f64,
    payoff: &str, direction: &str, product: &str,
    ki: Option<f64>, ko: Option<f64>, lots: f64, rebate: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(euro_barrier_amer_greeks_inner(
        OptionChar::from_str(char), tau, vol, k, s, r, payoff,
        Direction::from_str(direction), product, ki, ko, lots,
    ))
}

#[pyfunction]
#[pyo3(name = "euro_barrier_euro_greeks")]
#[pyo3(signature = (char, tau, vol, k, s, r, payoff, direction, product, ki=None, ko=None, lots=1.0, order=None, rebate=0.0, bvol=None, bvol2=None, dbarrier=None))]
pub fn py_euro_barrier_euro_greeks(
    char: &str, tau: f64, vol: f64, k: f64, s: f64, r: f64,
    payoff: &str, direction: &str, product: &str,
    ki: Option<f64>, ko: Option<f64>, lots: f64,
    order: Option<i32>, rebate: f64,
    bvol: Option<f64>, bvol2: Option<f64>, dbarrier: Option<f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(euro_barrier_euro_greeks_inner(
        OptionChar::from_str(char), tau, vol, k, s, r, payoff,
        direction, product, ki, ko, lots, bvol, bvol2, dbarrier,
    ))
}

#[pyfunction]
#[pyo3(name = "digital_greeks")]
pub fn py_digital_greeks(
    char: &str, k: f64, dbar: f64, tau: f64, vol: f64, vol2: f64,
    s: f64, r: f64, product: &str, payoff: &str, lots: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(digital_greeks_inner(
        OptionChar::from_str(char), k, dbar, tau, vol, vol2, s, r, product, payoff, lots,
    ))
}

#[pyfunction]
#[pyo3(name = "call_put_spread_greeks")]
#[pyo3(signature = (s, k1, k2, r, vol1, vol2, tau, optiontype, product, lots, payoff, b=0.0))]
pub fn py_call_put_spread_greeks(
    s: f64, k1: f64, k2: f64, r: f64, vol1: f64, vol2: f64,
    tau: f64, optiontype: &str, product: &str, lots: f64, payoff: &str, b: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(call_put_spread_greeks_inner(
        s, k1, k2, r, vol1, vol2, tau, optiontype, product, lots, payoff,
    ))
}

#[pyfunction]
#[pyo3(name = "greeks_scaled")]
pub fn py_greeks_scaled(
    delta1: f64, gamma1: f64, theta1: f64, vega1: f64,
    product: &str, lots: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(greeks_scaled_inner(delta1, gamma1, theta1, vega1, product, lots))
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tau() -> f64 {
        233.0 / 365.0 + 1.0 / 365.0
    }

    #[test]
    fn test_vanilla_greeks() {
        let vol = 0.22;
        let s = 387.750;
        let r = 0.0;
        let product = "C";
        let lots = 10.0;

        let strikes = [350.0, 360.0, 375.0, 387.75, 388.0, 389.0, 400.0, 420.0, 440.0,
                       440.0, 420.0, 400.0, 387.75, 388.0, 389.0, 350.0, 360.0, 370.0];
        let chars: Vec<OptionChar> = (0..9).map(|_| OptionChar::Call)
            .chain((0..9).map(|_| OptionChar::Put))
            .collect();
        let deltas = [
            7.4842859309570300, 6.9484395361432800, 6.0944898647578700,
            5.3509159016096900, 5.3363727454162100, 5.2782508035247300,
            4.6473987996563200, 3.5737686353737800, 2.6448749876143700,
            -7.3551250123856300, -6.4262313646262200, -5.3526012003436800,
            -4.649084098390310, -4.663627254583790, -4.721749196475270,
            -2.515714069042970, -3.051560463856720, -3.616375394010970,
        ];
        let gammas = [
            15.059813000513300, 16.548906492286400, 18.130051215724700,
            18.770786536726700, 18.776711054794000, 18.797880046671100,
            18.770083735887700, 17.626313297716400, 15.456046231256800,
            15.456046231256800, 17.626313297716400, 18.770083735887700,
            18.770786536726700, 18.776711054794000, 18.797880046671100,
            15.059813000513300, 16.548906492286400, 17.698718589721000,
        ];
        let thetas = [
            -23.266402976629900, -25.566952740978800, -28.009715496308800,
            -28.999608676209800, -29.008761660034500, -29.041466335424900,
            -28.998522895959400, -27.231474133378900, -23.878556794137300,
            -23.878556794137300, -27.231474133378900, -28.998522895959400,
            -28.999608676209800, -29.008761660034500, -29.041466335424900,
            -23.266402976629900, -25.566952740978800, -27.343335462690400,
        ];
        let vegas = [
            494.939845139217000, 543.878812853549000, 595.843038739661000,
            616.900766384828000, 617.095475313462000, 617.791192953584000,
            616.877668877681000, 579.287722473697000, 507.962026348011000,
            507.962026348011000, 579.287722473697000, 616.877668877681000,
            616.900766384828000, 617.095475313462000, 617.791192953584000,
            494.939845139217000, 543.878812853549000, 581.667318024504000,
        ];

        for i in 0..strikes.len() {
            let (d, g, t, v) = euro_vanilla_greeks_inner(
                chars[i], strikes[i], tau(), vol, s, r, product, lots,
            );
            // Tolerance: ~1e-3 relative error due to statrs vs scipy norm CDF/PDF differences
            assert!((d - deltas[i]).abs() < 1e-2, "delta[{}]: got {}, expected {}", i, d, deltas[i]);
            assert!((g - gammas[i]).abs() < 1e-2, "gamma[{}]: got {}, expected {}", i, g, gammas[i]);
            assert!((t - thetas[i]).abs() < 1e-2, "theta[{}]: got {}, expected {}", i, t, thetas[i]);
            assert!((v - vegas[i]).abs() < 1e-1, "vega[{}]: got {}, expected {}", i, v, vegas[i]);
        }
    }

    #[test]
    fn test_amer_barrier_greeks() {
        let s = 387.750;
        let vol = 0.22;
        let r = 0.0;
        let product = "C";
        let lots = 10.0;

        let chars = [OptionChar::Put, OptionChar::Put, OptionChar::Put, OptionChar::Put,
                     OptionChar::Call, OptionChar::Call, OptionChar::Call, OptionChar::Call];
        let strikes = [390.0, 450.0, 400.0, 400.0, 370.0, 380.0, 380.0, 380.0];
        let kis: [Option<f64>; 8] = [Some(380.0), Some(400.0), None, None, Some(390.0), Some(360.0), None, None];
        let kos: [Option<f64>; 8] = [None, None, Some(410.0), Some(370.0), None, None, Some(420.0), Some(370.0)];
        let directions = [Direction::Down, Direction::Up, Direction::Up, Direction::Down,
                          Direction::Up, Direction::Down, Direction::Up, Direction::Down];
        let expected_deltas = [
            -4.783986635992220, 8.465546983416060, -8.826913688153580, 0.104405133782848,
            6.417554302373670, -2.185933112297530, -0.157934175770702, 8.952093100500490,
        ];
        let expected_thetas = [
            -29.109439649076300, -29.153737444399800, -2.471758863475060, 0.615362469696290,
            -27.394820388902700, -22.028421202236400, 1.878320472499140, -2.213958487979010,
        ];
        let expected_vegas = [
            618.568709025737000, 619.585744363217000, 52.512848382684200, -13.103350453807200,
            582.136493086884000, 468.151697544542000, -39.969178254451000, 47.037403962064200,
        ];

        for i in 0..8 {
            let (d, _g, t, v) = euro_barrier_amer_greeks_inner(
                chars[i], tau(), vol, strikes[i], s, r,
                "amer", directions[i], product, kis[i], kos[i], lots,
            );
            assert!((d - expected_deltas[i]).abs() < 1e-3,
                    "delta[{}]: got {}, expected {}", i, d, expected_deltas[i]);
            assert!((t - expected_thetas[i]).abs() < 1e-3,
                    "theta[{}]: got {}, expected {}", i, t, expected_thetas[i]);
            assert!((v - expected_vegas[i]).abs() < 1e-3,
                    "vega[{}]: got {}, expected {}", i, v, expected_vegas[i]);
        }
    }

    #[test]
    fn test_euro_barrier_greeks() {
        let s = 387.750;
        let vol = 0.22;
        let bvol = 0.22;
        let product = "C";
        let lots = 10.0;

        let chars = [OptionChar::Call, OptionChar::Call, OptionChar::Put, OptionChar::Put];
        let directions = ["up", "up", "down", "down"];
        let kis: [Option<f64>; 4] = [None, Some(390.0), None, Some(385.0)];
        let kos: [Option<f64>; 4] = [Some(390.0), None, Some(380.0), None];
        let strikes = [350.0, 380.0, 395.0, 395.0];

        let expected_deltas = [-0.0557490887856027, 5.8001729954295600,
                               0.0045642948890046, -5.0722086291524400];
        let expected_gammas = [-3.130091906576830, 18.658738232493300,
                               -0.450432407726885, 19.037792844756400];
        let expected_thetas = [4.835782466210640, -28.826501546721800,
                               0.695887917834766, -29.412115548619900];
        let expected_vegas = [-102.870281553938000, 613.218305630264000,
                              -14.803433888480200, 625.675912579732000];

        for i in 0..4 {
            let (d, g, t, v) = euro_barrier_euro_greeks_inner(
                chars[i], tau(), vol, strikes[i], s, 0.0,
                "amer", directions[i], product, kis[i], kos[i], lots,
                Some(bvol), Some(bvol), None,
            );
            // Tolerance relaxed for statrs vs scipy numerical differences
            assert!((d - expected_deltas[i]).abs() < 1e-2,
                    "delta[{}]: got {}, expected {}", i, d, expected_deltas[i]);
            assert!((g - expected_gammas[i]).abs() < 1e-2,
                    "gamma[{}]: got {}, expected {}", i, g, expected_gammas[i]);
            assert!((t - expected_thetas[i]).abs() < 1e-2,
                    "theta[{}]: got {}, expected {}", i, t, expected_thetas[i]);
            assert!((v - expected_vegas[i]).abs() < 1e-1,
                    "vega[{}]: got {}, expected {}", i, v, expected_vegas[i]);
        }
    }
}
