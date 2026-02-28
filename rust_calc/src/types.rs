/// Option type: Call or Put
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionChar {
    Call,
    Put,
}

impl OptionChar {
    pub fn from_str(s: &str) -> Self {
        match s {
            "call" => OptionChar::Call,
            "put" => OptionChar::Put,
            _ => panic!("Invalid option type: '{}'. Expected 'call' or 'put'.", s),
        }
    }

    /// phi: +1 for call, -1 for put
    pub fn phi(&self) -> f64 {
        match self {
            OptionChar::Call => 1.0,
            OptionChar::Put => -1.0,
        }
    }
}

/// Barrier direction: Up or Down
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Up,
    Down,
}

impl Direction {
    pub fn from_str(s: &str) -> Self {
        match s {
            "up" => Direction::Up,
            "down" => Direction::Down,
            _ => panic!("Invalid direction: '{}'. Expected 'up' or 'down'.", s),
        }
    }

    /// eta: -1 for up, +1 for down
    pub fn eta(&self) -> f64 {
        match self {
            Direction::Up => -1.0,
            Direction::Down => 1.0,
        }
    }
}
