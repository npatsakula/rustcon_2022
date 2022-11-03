use std::{path::PathBuf, process::ExitCode};

use clap::Parser;
use evac::grammar::TopLevelExpressionParser;

#[derive(Debug, Parser)]
struct Opt {
    /// Show warnings.
    #[clap(short, long)]
    warnings: bool,

    /// Input file.
    #[clap(short, long)]
    input: Option<PathBuf>,
}

fn main() -> ExitCode {
    let Opt { input, .. } = Opt::parse();
    let source = if let Some(path) = input {
        std::io::read_to_string(std::fs::File::open(path).expect("Failed to open source file."))
            .expect("Failed to read file.")
    } else {
        std::io::read_to_string(std::io::stdin()).expect("Failed to read from STDIN.")
    };

    let expression = match TopLevelExpressionParser::new().parse(&mut Default::default(), &source) {
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
        Ok(expression) => expression,
    };

    println!("{}", expression.evaluate());

    ExitCode::SUCCESS
}
