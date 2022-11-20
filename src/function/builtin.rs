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

impl<'input> super::FunctionTrait<'input> for BuiltinFunction {
    fn name(&self) -> &'input str {
        match self {
            Self::Pi => "pi",
            Self::Cos => "cos",
            Self::Sin => "sin",
        }
    }

    fn arguments_count(&self) -> usize {
        match self {
            Self::Pi => 0,
            Self::Cos | Self::Sin => 1,
        }
    }

    fn evaluate(&self, arguments: &[Expression<f64>]) -> Result<f64, Error> {
        snafu::ensure!(
            arguments.len() == self.arguments_count(),
            WrongArgumentsAmountSnafu {
                name: self.name(),
                expected: self.arguments_count(),
                parsed: arguments.len(),
            }
        );

        Ok(match self {
            Self::Pi => std::f64::consts::PI,
            Self::Cos => arguments[0].evaluate().cos(),
            Self::Sin => arguments[0].evaluate().sin(),
        })
    }
}

impl BuiltinFunction {
    pub fn build_map() -> HashMap<&'static str, Self> {
        [("pi", Self::Pi), ("cos", Self::Cos), ("sin", Self::Sin)]
            .into_iter()
            .collect()
    }
}
