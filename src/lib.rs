use std::fmt::Display;

use lalrpop_util::lalrpop_mod;

pub mod function;
use function::{Context, Function};

pub mod error;
use error::*;
use serde::{Deserialize, Serialize};

pub mod block;

pub mod properties;
pub mod utils;

pub mod lexer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, Serialize, Deserialize)]
pub enum Operation {
    #[display(fmt = "+")]
    Plus,
    #[display(fmt = "*")]
    Product,
    #[display(fmt = "/")]
    Division,
    #[display(fmt = "-")]
    Minus,
}

impl Operation {
    #[inline(always)]
    pub fn evaluate(&self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Plus => lhs + rhs,
            Self::Product => lhs * rhs,
            Self::Division => lhs / rhs,
            Self::Minus => lhs - rhs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression<'input, V> {
    Value(V),

    Op {
        left: Box<Self>,
        op: Operation,
        right: Box<Self>,
    },

    FnCall {
        function: Function<'input>,
        arguments: Vec<Self>,
    },
}

impl<'input, V: Display> Display for Expression<'input, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Value(v) => v.fmt(f),
            Expression::Op { left, op, right } => write!(f, "{left} {op} {right}"),
            Expression::FnCall {
                function,
                arguments,
            } => {
                write!(f, "{}(", function.name())?;
                let mut source = arguments.iter();

                if let Some(first) = source.next() {
                    write!(f, "{first}")?;
                }

                for arg in source {
                    write!(f, ", {arg}")?;
                }

                write!(f, ")")
            }
        }
    }
}

impl<'input, V> Expression<'input, V> {
    pub fn op(lhs: Self, op: Operation, rhs: Self) -> Self {
        Self::Op {
            left: lhs.into(),
            op,
            right: rhs.into(),
        }
    }

    pub fn function_call(
        context: &Context<'input>,
        name: &'input str,
        arguments: Vec<Self>,
    ) -> Result<Self, Error> {
        let function = context.get_function(name)?;

        let expected = function.arguments_count();
        let parsed = arguments.len();

        snafu::ensure!(
            expected == parsed,
            WrongArgumentsAmountSnafu {
                name,
                expected,
                parsed
            }
        );
        Ok(Self::FnCall {
            function,
            arguments,
        })
    }

    pub fn leafs_count(&self) -> usize {
        match self {
            Self::Value(_) => 1,
            Self::Op { left, right, .. } => 1 + left.leafs_count() + right.leafs_count(),
            Self::FnCall { arguments, .. } => arguments.iter().map(Self::leafs_count).sum(),
        }
    }
}

impl Expression<'_, f64> {
    pub fn evaluate(&self) -> f64 {
        match self {
            Self::Value(v) => *v,
            Self::Op { left, op, right } => op.evaluate(left.evaluate(), right.evaluate()),
            Self::FnCall {
                function,
                arguments,
            } => function.evaluate(arguments),
        }
    }
}

pub mod prelude {
    pub use crate::error::*;
    pub use crate::function::{BuiltinFunction, FunctionValue, UserDefinedFunction};
    pub use crate::{Expression, Operation};
}

lalrpop_mod!(
    #[cfg_attr(not(test), allow(dead_code, unused_imports))]
    #[allow(
        clippy::clone_on_copy,
        clippy::needless_lifetimes,
        clippy::just_underscores_and_digits,
        clippy::too_many_arguments,
    )]
    pub grammar
);

#[cfg(test)]
pub mod parse_test {
    use super::Context;
    use crate::grammar::TopLevelExpressionParser;
    use crate::lexer::EvacLexer;
    use crate::properties::expression;

    use proptest::prelude::*;
    use test_case::test_case;

    proptest! {
        #[test]
        fn parse(expr in expression().prop_filter("Result must be comparable.", |e| e.evaluate().is_normal())) {
            let evaluate_source = expr.evaluate();
            let stringified = format!("{expr}");
            let lexer = EvacLexer::new(&stringified);
            let parsed = TopLevelExpressionParser::new().parse(&mut Context::default(), lexer).unwrap();
            let evaluate_parsed = parsed.evaluate();
            prop_assert_eq!(evaluate_source, evaluate_parsed);
        }
    }

    #[test_case(include_str!("../data/expression_1.data") => matches Ok(_); "expression 1")]
    fn test_data_parse(source: &'static str) -> Result<f64, String> {
        let lexer = EvacLexer::new(source);
        TopLevelExpressionParser::new()
            .parse(&mut Context::default(), lexer)
            .map_err(|e| e.to_string())
            .map(|e| e.evaluate())
    }
}
