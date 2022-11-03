use std::fmt::Display;

use lalrpop_util::lalrpop_mod;

pub mod function;
use function::{Context, Function};

pub mod error;
use error::*;
use serde::{Deserialize, Serialize};

pub mod properties;
pub mod utils;

#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, Serialize, Deserialize)]
pub enum Operation {
    #[display(fmt = "+")]
    Sum,
    #[display(fmt = "*")]
    Mul,
    #[display(fmt = "/")]
    Div,
    #[display(fmt = "-")]
    Sub,
}

impl Operation {
    #[inline(always)]
    pub fn evaluate(&self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Sum => lhs + rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
            Self::Sub => lhs - rhs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression<V> {
    Value(V),

    Op {
        left: Box<Self>,
        op: Operation,
        right: Box<Self>,
    },

    FnCall {
        function: Function,
        arguments: Vec<Self>,
    },
}

impl<V: Display> Display for Expression<V> {
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

impl<V> Expression<V> {
    pub fn op(lhs: Self, op: Operation, rhs: Self) -> Self {
        Self::Op {
            left: lhs.into(),
            op,
            right: rhs.into(),
        }
    }

    pub fn function_call(
        context: &Context,
        name: String,
        arguments: Vec<Self>,
    ) -> Result<Self, Error> {
        let function = context.get_function(&name)?;

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

impl Expression<f64> {
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
    pub use crate::function::{BuiltinFunction, Context, FunctionValue, UserDefinedFunction};
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
    use crate::properties::top_level_expression;

    use proptest::prelude::*;
    use test_case::test_case;

    proptest! {
        // Мы хотим сгенерировать тысячу деревьев:
        #![proptest_config(ProptestConfig::with_cases(1_000))]
        // Запускаться это будет как обычный тест:
        #[test]
        // Результат вычисления должен быть сравним, потому мы добавим фильтр,
        // который проверяет, что он является нормальным значением.
        fn parse(expr in top_level_expression().prop_filter("Result must be comparable.", |e| e.evaluate().is_normal())) {
            // Вычисляем выражение:
            let evaluate_source = expr.evaluate();
            // Переводим выражение в строку:
            let stringified = format!("{expr}");
            // Разбираем выражение из строки обратно:
            let parsed = TopLevelExpressionParser::new().parse(&mut Context::from_expression(&expr), &stringified).unwrap();
            // Вычисляем выражение, которое разобралось из строки:
            let evaluate_parsed = parsed.evaluate();
            // Сравниваем, что результаты вычисления равны:
            prop_assert_eq!(evaluate_source, evaluate_parsed);
        }
    }

    #[test_case(include_str!("../data/expression_1.data") => matches Ok(_); "expression 1")]
    fn test_data_parse(source: &'static str) -> Result<f64, String> {
        TopLevelExpressionParser::new()
            .parse(&mut Context::default(), source)
            .map_err(|e| e.to_string())
            .map(|e| e.evaluate())
    }
}
