pub mod user_defined;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
pub use user_defined::{FunctionValue, UserDefinedFunction};

pub mod builtin;
pub use builtin::BuiltinFunction;

use crate::{error::*, Expression};

use std::collections::HashMap;
use std::fmt::Display;
use std::sync::Arc;

use snafu::OptionExt;

#[derive(Debug, Clone)]
pub struct Context<'input> {
    builtin_functions: HashMap<&'static str, BuiltinFunction>,
    user_defined_functions: HashMap<&'input str, Arc<UserDefinedFunction<'input>>>,
}

impl<'i> Default for Context<'i> {
    fn default() -> Self {
        Self {
            builtin_functions: BuiltinFunction::build_map(),
            user_defined_functions: Default::default(),
        }
    }
}

impl Display for Context<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for function in self.sort_calls() {
            writeln!(f, "{function}")?;
        }

        Ok(())
    }
}

impl<'input> Context<'input> {
    pub fn sort_calls(&self) -> Vec<Arc<UserDefinedFunction<'input>>> {
        fn depends<'i>(
            expression: &'i Expression<FunctionValue<'i>>,
            result: &mut Vec<Arc<UserDefinedFunction<'i>>>,
        ) {
            match expression {
                Expression::Value(_) => (),
                Expression::Op { left, right, .. } => {
                    depends(left, result);
                    depends(right, result);
                }
                Expression::FnCall {
                    function,
                    arguments,
                } => {
                    arguments.iter().for_each(|a| depends(a, result));
                    if let Function::UserDefined(f) = function {
                        if !result.contains(f) {
                            result.push(f.clone());
                            depends(&f.expression, result);
                        }
                    }
                }
            }
        }

        let mut result = self
            .user_defined_functions
            .values()
            .map(|f| {
                let mut result = Vec::new();
                depends(&f.expression, &mut result);
                (f, result.len())
            })
            .collect::<Vec<_>>();

        result.sort_by_key(|(_, count)| *count);
        result.into_iter().map(|(f, _)| f).cloned().collect()
    }

    #[cfg_attr(not(test), allow(dead_code, unused_imports))]
    pub fn from_expression<V>(source: &'input Expression<'input, V>) -> Self {
        fn helper<'i, V>(source: &'i Expression<'i, V>, context: &mut Context<'i>) {
            match source {
                Expression::Value(_) => (),
                Expression::Op { left, right, .. } => {
                    helper(left.as_ref(), context);
                    helper(right.as_ref(), context);
                }
                Expression::FnCall {
                    arguments,
                    function,
                } => {
                    arguments.iter().for_each(|a| helper(a, context));
                    if let Function::UserDefined(f) = function {
                        context
                            .user_defined_functions
                            .insert(function.name(), f.clone());
                        helper(&f.expression, context);
                    }
                }
            }
        }

        let mut result = Self::default();
        helper(source, &mut result);
        result
    }

    fn function_names(&self) -> impl Iterator<Item = &'input str> + '_ {
        self.builtin_functions
            .keys()
            .copied()
            .chain(self.user_defined_functions.keys().copied())
    }

    pub fn get_function(&self, name: &'input str) -> Result<Function<'input>, Error> {
        self.builtin_functions
            .get(name)
            .map(|f| Function::from(*f))
            .or_else(|| {
                self.user_defined_functions
                    .get(name)
                    .cloned()
                    .map(Into::<Function>::into)
            })
            .with_context(|| UndefinedFunctionSnafu {
                name,
                options: self
                    .function_names()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>(),
            })
    }

    pub fn add_function(
        &mut self,
        name: &'input str,
        function: UserDefinedFunction<'input>,
    ) -> Result<(), Error> {
        if self.user_defined_functions.contains_key(name) {
            return FunctionAlreadyExistSnafu { name }.fail();
        }

        self.user_defined_functions.insert(name, function.into());
        Ok(())
    }

    pub fn clear(&mut self) {
        self.user_defined_functions.clear();
    }
}

#[derive(Debug, Clone, PartialEq, derive_more::From, Serialize, Deserialize)]
pub enum Function<'input> {
    Builtin(BuiltinFunction),

    #[serde(borrow)]
    #[serde(serialize_with = "serialize_user_defined")]
    #[serde(deserialize_with = "deserialize_user_defined")]
    UserDefined(Arc<UserDefinedFunction<'input>>),
}

fn deserialize_user_defined<'de, D>(
    deserializer: D,
) -> Result<Arc<UserDefinedFunction<'de>>, D::Error>
where
    D: Deserializer<'de>,
{
    UserDefinedFunction::deserialize(deserializer).map(Arc::new)
}

fn serialize_user_defined<S>(f: &Arc<UserDefinedFunction>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    f.as_ref().serialize(serializer)
}

impl<'input> Function<'input> {
    pub fn name(&self) -> &str {
        match self {
            Function::Builtin(b) => b.name(),
            Function::UserDefined(u) => u.name,
        }
    }

    pub fn arguments_count(&self) -> usize {
        match self {
            Function::Builtin(b) => b.args_count(),
            Function::UserDefined(u) => u.argument_count(),
        }
    }

    pub fn evaluate(&self, arguments: &[Expression<f64>]) -> f64 {
        match self {
            Function::Builtin(b) => b.evaluate(arguments).unwrap(),
            Function::UserDefined(u) => u.evaluate(arguments).unwrap(),
        }
    }
}
