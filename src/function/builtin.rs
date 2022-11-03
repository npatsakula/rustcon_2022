use std::collections::HashMap;

use serde::Deserialize;
use serde::Serialize;

use crate::error::*;
use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, proptest_derive::Arbitrary)]
pub enum BuiltinFunction {
    Pi,
    Cos,
    Sin,
}

impl BuiltinFunction {
    pub fn build_map() -> HashMap<&'static str, Self> {
        [("pi", Self::Pi), ("cos", Self::Cos), ("sin", Self::Sin)]
            .into_iter()
            .collect()
    }

    pub const fn args_count(&self) -> usize {
        match self {
            Self::Pi => 0,
            Self::Cos | Self::Sin => 1,
        }
    }

    pub const fn name(&self) -> &str {
        match self {
            Self::Pi => "pi",
            Self::Cos => "cos",
            Self::Sin => "sin",
        }
    }

    pub fn evaluate(&self, args: &[Expression<f64>]) -> Result<f64, Error> {
        snafu::ensure!(
            args.len() == self.args_count(),
            WrongArgumentsAmountSnafu {
                name: self.name(),
                expected: self.args_count(),
                parsed: args.len(),
            }
        );

        Ok(match self {
            BuiltinFunction::Pi => std::f64::consts::PI,
            BuiltinFunction::Cos => args[0].evaluate().cos(),
            BuiltinFunction::Sin => args[0].evaluate().sin(),
        })
    }
}
