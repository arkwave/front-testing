use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy)]
pub struct Multipliers {
    pub dollar_mult: f64,
    pub lot_mult: f64,
    pub futures_tick: f64,
    pub options_tick: f64,
    pub pnl_mult: f64,
}

impl Multipliers {
    pub const fn new(dollar_mult: f64, lot_mult: f64, futures_tick: f64, options_tick: f64, pnl_mult: f64) -> Self {
        Self { dollar_mult, lot_mult, futures_tick, options_tick, pnl_mult }
    }
}

/// Static multipliers table matching Python's `multipliers` dict in calc.py
pub static MULTIPLIERS: LazyLock<HashMap<&'static str, Multipliers>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("LH",  Multipliers::new(22.046, 18.143881, 0.025, 1.0, 400.0));
    m.insert("LSU", Multipliers::new(1.0, 50.0, 0.1, 10.0, 50.0));
    m.insert("QC",  Multipliers::new(1.2153, 10.0, 1.0, 25.0, 12.153));
    m.insert("SB",  Multipliers::new(22.046, 50.802867, 0.01, 0.25, 1120.0));
    m.insert("CC",  Multipliers::new(1.0, 10.0, 1.0, 50.0, 10.0));
    m.insert("CT",  Multipliers::new(22.046, 22.679851, 0.01, 1.0, 500.0));
    m.insert("KC",  Multipliers::new(22.046, 17.009888, 0.05, 2.5, 375.0));
    m.insert("W",   Multipliers::new(0.3674333, 136.07911, 0.25, 10.0, 50.0));
    m.insert("S",   Multipliers::new(0.3674333, 136.07911, 0.25, 10.0, 50.0));
    m.insert("C",   Multipliers::new(0.393678571428571, 127.007166832986, 0.25, 10.0, 50.0));
    m.insert("BO",  Multipliers::new(22.046, 27.215821, 0.01, 0.5, 600.0));
    m.insert("LC",  Multipliers::new(22.046, 18.143881, 0.025, 1.0, 400.0));
    m.insert("LRC", Multipliers::new(1.0, 10.0, 1.0, 50.0, 10.0));
    m.insert("KW",  Multipliers::new(0.3674333, 136.07911, 0.25, 10.0, 50.0));
    m.insert("SM",  Multipliers::new(1.1023113, 90.718447, 0.1, 5.0, 100.0));
    m.insert("COM", Multipliers::new(1.0604, 50.0, 0.25, 2.5, 53.02));
    m.insert("CA",  Multipliers::new(1.0604, 50.0, 0.25, 1.0, 53.02));
    m.insert("MW",  Multipliers::new(0.3674333, 136.07911, 0.25, 10.0, 50.0));
    m
});

pub fn get_multipliers(product: &str) -> &Multipliers {
    MULTIPLIERS.get(product).unwrap_or_else(|| panic!("Unknown product: '{}'", product))
}
