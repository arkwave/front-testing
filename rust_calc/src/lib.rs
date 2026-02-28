#![allow(unused_variables)]

mod types;
mod multipliers;
mod bsm;
mod barrier_amer;
mod barrier_euro;
mod greeks;
mod iv;

use pyo3::prelude::*;

#[pymodule]
fn rust_calc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pricing functions
    m.add_function(wrap_pyfunction!(bsm::bsm_euro, m)?)?;
    m.add_function(wrap_pyfunction!(barrier_amer::barrier_amer, m)?)?;
    m.add_function(wrap_pyfunction!(barrier_euro::barrier_euro, m)?)?;
    m.add_function(wrap_pyfunction!(barrier_euro::py_digital_option, m)?)?;
    m.add_function(wrap_pyfunction!(barrier_euro::py_call_put_spread, m)?)?;
    m.add_function(wrap_pyfunction!(bsm::compute_value, m)?)?;

    // Greeks functions
    m.add_function(wrap_pyfunction!(greeks::py_compute_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(greeks::py_euro_vanilla_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(greeks::py_euro_barrier_amer_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(greeks::py_euro_barrier_euro_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(greeks::py_digital_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(greeks::py_call_put_spread_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(greeks::py_greeks_scaled, m)?)?;

    // IV functions
    m.add_function(wrap_pyfunction!(iv::py_newton_raphson, m)?)?;
    m.add_function(wrap_pyfunction!(iv::py_compute_iv, m)?)?;
    m.add_function(wrap_pyfunction!(iv::py_compute_strike_from_delta, m)?)?;

    Ok(())
}
