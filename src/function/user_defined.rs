use std::{collections::HashMap, fmt::Display};

use serde::{Serialize, Deserialize};

use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, derive_more::From, derive_more::Display, Serialize, Deserialize)]
pub enum FunctionValue {
    Float(f64),
    Variable(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserDefinedFunction {
    pub(crate) name: String,
    pub(crate) args: Vec<String>,
    pub(crate) expression: Expression<FunctionValue>,
}

impl Display for UserDefinedFunction {
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

impl UserDefinedFunction {
    fn build_frequency_map(source: &[String]) -> HashMap<&str, usize> {
        source
            .iter()
            .fold(HashMap::with_capacity(source.len()), |mut map, arg| {
                *map.entry(arg.as_str()).or_insert(0) += 1;
                map
            })
    }

    fn check_duplicate(source: &[String]) -> Result<(), Error> {
        for (arg, n) in Self::build_frequency_map(source) {
            snafu::ensure!(n == 1, ArgDuplicateSnafu { arg, n });
        }
        Ok(())
    }

    fn check_binds(args: &[String], expression: &Expression<FunctionValue>) -> Result<(), Error> {
        let check_binds = |expr| Self::check_binds(args, expr);
        match expression {
            Expression::Value(FunctionValue::Float(_)) => (),
            Expression::Value(FunctionValue::Variable(name)) => snafu::ensure!(
                args.contains(name),
                UnboundedVariableSnafu {
                    name,
                    options: args
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
        name: String,
        args: Vec<String>,
        expression: Expression<FunctionValue>,
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

    fn evaluate_helper(expression: &Expression<FunctionValue>, args: &HashMap<&str, f64>) -> f64 {
        let evaluate_helper = |expr| Self::evaluate_helper(expr, args);
        match expression {
            Expression::Value(FunctionValue::Float(v)) => *v,
            Expression::Value(FunctionValue::Variable(name)) => *args.get(name.as_str()).unwrap(),
            Expression::Op { left, op, right } => {
                op.evaluate(evaluate_helper(left), evaluate_helper(right))
            }
            Expression::FnCall {
                function,
                arguments,
            } => {
                let arguments = arguments
                    .iter()
                    .map(evaluate_helper)
                    .map(Expression::Value)
                    .collect::<Vec<_>>();

                function.evaluate(&arguments)
            }
        }
    }

    pub fn evaluate(&self, arguments: &[Expression<f64>]) -> Result<f64, Error> {
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
            .map(|(name, expr)| (name.as_str(), expr.evaluate()))
            .collect::<HashMap<_, _>>();

        Ok(Self::evaluate_helper(&self.expression, &map))
    }
}
