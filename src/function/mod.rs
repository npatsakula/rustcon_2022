pub mod user_defined;
use serde::{Deserialize, Serialize};
pub use user_defined::{FunctionValue, UserDefinedFunction};

pub mod builtin;
pub use builtin::BuiltinFunction;

use crate::{error::*, Expression};

use std::collections::HashMap;
use std::sync::Arc;

use snafu::OptionExt;

#[derive(Debug)]
pub struct Context<'input> {
    builtin_functions: HashMap<&'static str, BuiltinFunction>,
    user_defined_functions: HashMap<&'input str, Arc<UserDefinedFunction<'input>>>,
}

impl Default for Context<'static> {
    fn default() -> Self {
        Self {
            builtin_functions: BuiltinFunction::build_map(),
            user_defined_functions: Default::default(),
        }
    }
}

impl<'input> Context<'input> {
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
    #[serde(skip)]
    UserDefined(Arc<UserDefinedFunction<'input>>),
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
