use pyo3::prelude::*;

use crate::bsm::{bsm_euro_inner, norm_cdf};
use crate::types::{Direction, OptionChar};

/// Haug helper A/B: used in closed-form American barrier pricing.
fn a_b(flag: char, phi: f64, b: f64, r: f64, x1: f64, x2: f64, tau: f64, vol: f64, s: f64, k: f64) -> f64 {
    let x = if flag == 'A' { x1 } else { x2 };
    let sqrt_tau = tau.sqrt();
    phi * s * ((b - r) * tau).exp() * norm_cdf(phi * x)
        - phi * k * (-r * tau).exp() * norm_cdf(phi * x - phi * vol * sqrt_tau)
}

/// Haug helper C/D
fn c_d(flag: char, phi: f64, s: f64, b: f64, r: f64, tau: f64, h: f64, mu: f64, eta: f64, y1: f64, y2: f64, k: f64, vol: f64) -> f64 {
    let y = if flag == 'C' { y1 } else { y2 };
    let sqrt_tau = tau.sqrt();
    phi * s * ((b - r) * tau).exp() * (h / s).powf(2.0 * (mu + 1.0)) * norm_cdf(eta * y)
        - phi * k * (-r * tau).exp() * (h / s).powf(2.0 * mu) * norm_cdf(eta * y - eta * vol * sqrt_tau)
}

/// Haug helper E
fn e_f(rebate: f64, r: f64, tau: f64, eta: f64, x2: f64, vol: f64, h: f64, s: f64, mu: f64, y2: f64) -> f64 {
    let sqrt_tau = tau.sqrt();
    rebate * (-r * tau).exp()
        * (norm_cdf(eta * x2 - eta * vol * sqrt_tau)
           - (h / s).powf(2.0 * mu) * norm_cdf(eta * y2 - eta * vol * sqrt_tau))
}

/// Haug helper F
fn f_f(rebate: f64, h: f64, s: f64, mu: f64, lambd: f64, eta: f64, z: f64, vol: f64, tau: f64) -> f64 {
    let sqrt_tau = tau.sqrt();
    rebate * ((h / s).powf(mu + lambd) * norm_cdf(eta * z)
              + (h / s).powf(mu - lambd) * norm_cdf(eta * z - 2.0 * eta * lambd * vol * sqrt_tau))
}

/// American barrier option pricing (Haug closed-form).
///
/// 16-branch decision tree covering all call/put × up/down × KI/KO combinations.
/// Equivalent to Python `_barrier_amer`.
pub fn barrier_amer_inner(
    char: OptionChar,
    tau: f64,
    vol: f64,
    k: f64,
    s: f64,
    r: f64,
    _payoff: &str,
    direction: Direction,
    ki: Option<f64>,
    ko: Option<f64>,
    rebate: f64,
) -> f64 {
    let vol = if vol == 0.0 { 0.0001 } else { vol };
    assert!(tau > 0.0, "tau must be positive for barrier pricing");

    let eta = direction.eta();
    let phi = char.phi();
    let b = 0.0;
    let mu = (b - (vol * vol / 2.0)) / (vol * vol);
    let lambd = (mu * mu + 2.0 * r / (vol * vol)).sqrt();
    let h = ki.or(ko).expect("Either ki or ko must be provided");

    let sqrt_tau = tau.sqrt();
    let vol_sqrt = vol * sqrt_tau;

    let x1 = (s / k).ln() / vol_sqrt + (1.0 + mu) * vol_sqrt;
    let x2 = (s / h).ln() / vol_sqrt + (1.0 + mu) * vol_sqrt;
    let y1 = (h * h / (s * k)).ln() / vol_sqrt + (1.0 + mu) * vol_sqrt;
    let y2 = (h / s).ln() / vol_sqrt + (1.0 + mu) * vol_sqrt;
    let z = (h / s).ln() / vol_sqrt + lambd * vol_sqrt;

    let a = a_b('A', phi, b, r, x1, x2, tau, vol, s, k);
    let b_val = a_b('B', phi, b, r, x1, x2, tau, vol, s, k);
    let c = c_d('C', phi, s, b, r, tau, h, mu, eta, y1, y2, k, vol);
    let d = c_d('D', phi, s, b, r, tau, h, mu, eta, y1, y2, k, vol);
    let e = e_f(rebate, r, tau, eta, x2, vol, h, s, mu, y2);
    let f = f_f(rebate, h, s, mu, lambd, eta, z, vol, tau);

    let vanilla = || bsm_euro_inner(char, tau, vol, k, s, r);
    let rebate_pv = rebate * (-r * tau).exp();

    match (char, direction) {
        // ── Call Up ──
        (OptionChar::Call, Direction::Up) => {
            if let Some(ki_val) = ki {
                // Call Up In
                if s >= ki_val {
                    vanilla()
                } else if k >= ki_val && tau > 0.0 {
                    a + e
                } else if k >= ki_val {
                    0.0
                } else if k < ki_val && tau > 0.0 {
                    b_val - c + d + e
                } else {
                    0.0
                }
            } else {
                let ko_val = ko.unwrap();
                // Call Up Out
                if s >= ko_val {
                    rebate_pv
                } else if k >= ko_val && tau > 0.0 {
                    f
                } else if k >= ko_val {
                    vanilla()
                } else if k < ko_val && tau > 0.0 {
                    a - b_val + c - d + f
                } else {
                    vanilla()
                }
            }
        }
        // ── Call Down ──
        (OptionChar::Call, Direction::Down) => {
            if let Some(ki_val) = ki {
                // Call Down In
                if s <= ki_val {
                    vanilla()
                } else if k >= ki_val && tau > 0.0 {
                    c + e
                } else if k >= ki_val {
                    0.0
                } else if k < ki_val && tau > 0.0 {
                    a - b_val + d + e
                } else {
                    0.0
                }
            } else {
                let ko_val = ko.unwrap();
                // Call Down Out
                if s < ko_val {
                    rebate_pv
                } else if k >= ko_val && tau > 0.0 {
                    a - c + f
                } else if k >= ko_val {
                    vanilla()
                } else if k < ko_val && tau > 0.0 {
                    b_val - d + f
                } else {
                    vanilla()
                }
            }
        }
        // ── Put Up ──
        (OptionChar::Put, Direction::Up) => {
            if let Some(ki_val) = ki {
                // Put Up In
                if s >= ki_val {
                    vanilla()
                } else if k >= ki_val && tau > 0.0 {
                    a - b_val + d + e
                } else if k >= ki_val {
                    0.0
                } else if k < ki_val && tau > 0.0 {
                    c + e
                } else {
                    0.0
                }
            } else {
                let ko_val = ko.unwrap();
                // Put Up Out
                if s >= ko_val {
                    rebate_pv
                } else if k >= ko_val && tau > 0.0 {
                    b_val - d + f
                } else if k >= ko_val {
                    vanilla()
                } else if k < ko_val && tau > 0.0 {
                    a - c + f
                } else {
                    vanilla()
                }
            }
        }
        // ── Put Down ──
        (OptionChar::Put, Direction::Down) => {
            if let Some(ki_val) = ki {
                // Put Down In
                if s <= ki_val {
                    vanilla()
                } else if k >= ki_val && tau > 0.0 {
                    b_val - c + d + e
                } else if k >= ki_val {
                    0.0
                } else if k < ki_val && tau > 0.0 {
                    a + e
                } else {
                    0.0
                }
            } else {
                let ko_val = ko.unwrap();
                // Put Down Out
                if s <= ko_val {
                    rebate_pv
                } else if k > ko_val && tau > 0.0 {
                    a - b_val + c - d + f
                } else if k > ko_val {
                    vanilla()
                } else if k < ko_val && tau > 0.0 {
                    f
                } else {
                    vanilla()
                }
            }
        }
    }
}

// ── Python-exposed function ───────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "barrier_amer")]
#[pyo3(signature = (char, tau, vol, k, s, r, payoff, direction, ki=None, ko=None, rebate=0.0))]
pub fn barrier_amer(
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
    rebate: f64,
) -> PyResult<f64> {
    Ok(barrier_amer_inner(
        OptionChar::from_str(char),
        tau, vol, k, s, r, payoff,
        Direction::from_str(direction),
        ki, ko, rebate,
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
    fn test_amer_barrier_pricing() {
        let s = 387.750;
        let vol = 0.22;
        let r = 0.0;
        let chars = [OptionChar::Put, OptionChar::Put, OptionChar::Put, OptionChar::Put,
                     OptionChar::Call, OptionChar::Call, OptionChar::Call, OptionChar::Call];
        let strikes = [390.0, 450.0, 400.0, 400.0, 370.0, 380.0, 380.0, 380.0];
        let kis: [Option<f64>; 8] = [Some(380.0), Some(400.0), None, None, Some(390.0), Some(360.0), None, None];
        let kos: [Option<f64>; 8] = [None, None, Some(410.0), Some(370.0), None, None, Some(420.0), Some(370.0)];
        let directions = [Direction::Down, Direction::Up, Direction::Up, Direction::Down,
                          Direction::Up, Direction::Down, Direction::Up, Direction::Down];
        let prices = [
            28.428865379491300, 50.449884052880000, 19.558953734663000, 0.198835317329896,
            36.390665957702900, 9.182972317226490, 0.643093633583953, 15.841689331908300,
        ];

        for i in 0..8 {
            let val = barrier_amer_inner(
                chars[i], tau(), vol, strikes[i], s, r,
                "amer", directions[i], kis[i], kos[i], 0.0,
            );
            assert!(
                (val - prices[i]).abs() < 1e-8,
                "Case {}: got {}, expected {}", i, val, prices[i]
            );
        }
    }
}
