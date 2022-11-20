use std::fmt::Display;

use inkwell::execution_engine::FunctionLookupError;
use lalrpop_util::ParseError;
use snafu::Location;

#[derive(Debug, snafu::Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("Argument '{arg}' detected {n} times."))]
    ArgDuplicate {
        arg: String,
        n: usize,
        location: Location,
    },

    #[snafu(display("Unbound variable: {name}. Options: {options:?}."))]
    UnboundedVariable {
        name: String,
        options: Vec<String>,
        location: Location,
    },

    #[snafu(display("Function '{name}' is undefined. Possible options: {options:?}"))]
    UndefinedFunction {
        name: String,
        options: Vec<String>,
        location: Location,
    },

    #[snafu(display("Function '{name}' exist but required different from {parsed} amount of arguments: {expected}."))]
    WrongArgumentsAmount {
        name: String,
        expected: usize,
        parsed: usize,
    },

    #[snafu(display("Function '{name}' already exist."))]
    FunctionAlreadyExist { name: String, location: Location },

    #[snafu(display("Failed to parse expression: {message}"))]
    Parse { message: String, location: Location },

    #[snafu(display("Failed to lookup function '{function}' in module '{module}': {source}"))]
    FunctionLookup {
        source: FunctionLookupError,
        function: String,
        module: String,
        location: Location,
    }
}

impl<L: Display, T: Display, E: Display> From<ParseError<L, T, E>> for Error {
    fn from(source: ParseError<L, T, E>) -> Self {
        ParseSnafu {
            message: source.to_string(),
        }
        .build()
    }
}
