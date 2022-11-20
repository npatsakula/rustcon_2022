use derive_more::{Display, From};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Display};

use crate::prelude::*;

use super::FunctionTrait;

#[derive(Debug, Clone, PartialEq, From, Display, Serialize, Deserialize)]
pub enum FunctionValue<'input> {
    Float(f64),
    Variable(&'input str),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserDefinedFunction<'input, 'ctx> {
    pub(crate) name: &'input str,
    pub(crate) args: Vec<&'input str>,
    #[serde(borrow)]
    pub(crate) expression: Expression<'input, 'ctx, FunctionValue<'input>>,
}

impl<'input, 'jit> Display for UserDefinedFunction<'input, 'jit> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn {}(", self.name)?;

        let mut args = self.args.iter();
        if let Some(first) = args.next() {
            write!(f, "{first}")?;
        }

        for arg in args {
            write!(f, ", {arg}")?;
        }

        write!(f, ") = {};", self.expression)
    }
}

impl<'input, 'jit> UserDefinedFunction<'input, 'jit> {
    fn build_frequency_map(source: &[&'input str]) -> HashMap<&'input str, usize> {
        source
            .iter()
            .fold(HashMap::with_capacity(source.len()), |mut map, arg| {
                *map.entry(arg).or_insert(0) += 1;
                map
            })
    }

    fn check_duplicate(source: &[&'input str]) -> Result<(), Error> {
        for (arg, n) in Self::build_frequency_map(source) {
            snafu::ensure!(n == 1, ArgDuplicateSnafu { arg, n });
        }
        Ok(())
    }

    fn check_binds(
        args: &[&'input str],
        expression: &Expression<FunctionValue>,
    ) -> Result<(), Error> {
        let check_binds = |expr| Self::check_binds(args, expr);
        match expression {
            Expression::Value(FunctionValue::Float(_)) => (),
            Expression::Value(FunctionValue::Variable(name)) => snafu::ensure!(
                args.contains(name),
                UnboundedVariableSnafu {
                    name: name.to_string(),
                    options: args.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                }
            ),
            Expression::Op { left, right, .. } => {
                check_binds(left)?;
                check_binds(right)?;
            }
            Expression::FnCall {
                arguments,
                ..
                // function,
            } => {
                for arg in arguments {
                    check_binds(arg)?;
                }
            }
        }
        Ok(())
    }

    pub fn new(
        name: &'input str,
        args: Vec<&'input str>,
        expression: Expression<'input, 'jit, FunctionValue<'input>>,
    ) -> Result<Self, Error> {
        Self::check_duplicate(&args)?;
        Self::check_binds(&args, &expression)?;
        Ok(Self {
            name,
            args,
            expression,
        })
    }

    pub fn argument_count(&self) -> usize {
        self.args.len()
    }

    fn evaluate_helper(
        expression: &Expression<FunctionValue>,
        args: &HashMap<&str, f64>,
    ) -> Result<f64, Error> {
        let evaluate_helper = |expr| Self::evaluate_helper(expr, args);
        match expression {
            Expression::Value(FunctionValue::Float(v)) => Ok(*v),
            Expression::Value(FunctionValue::Variable(name)) => Ok(*args.get(name).unwrap()),
            Expression::Op { left, op, right } => {
                Ok(op.evaluate(evaluate_helper(left)?, evaluate_helper(right)?))
            }
            Expression::FnCall {
                function,
                arguments,
            } => {
                let arguments: Vec<_> = arguments
                    .iter()
                    .map(|e| evaluate_helper(e).map(Expression::Value))
                    .try_collect()?;

                function.evaluate(&arguments)
            }
        }
    }
}

impl<'input, 'ctx> FunctionTrait for UserDefinedFunction<'input, 'ctx> {
    fn name(&self) -> &str {
        self.name
    }

    fn arguments_count(&self) -> usize {
        self.args.len()
    }

    fn evaluate(&self, arguments: &[Expression<f64>]) -> Result<f64, Error> {
        snafu::ensure!(
            arguments.len() == self.argument_count(),
            WrongArgumentsAmountSnafu {
                name: self.name.to_string(),
                expected: self.argument_count(),
                parsed: arguments.len(),
            }
        );

        let map = self
            .args
            .iter()
            .zip(arguments.iter())
            .map(|(name, expr)| (*name, expr.evaluate()))
            .collect::<HashMap<_, _>>();

        Self::evaluate_helper(&self.expression, &map)
    }
}
