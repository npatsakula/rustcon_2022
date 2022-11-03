use std::{io::Read, path::Path, sync::Mutex};

use evac::{function::Context, grammar::TopLevelExpressionParser, Expression};
use once_cell::sync::Lazy;

#[global_allocator]
static ALLOC: mimalloc_rust::GlobalMiMalloc = mimalloc_rust::GlobalMiMalloc;

const DATASET_PATH: &str = "./data/expressions_30kk.bin.lz4";

static DATASET: Lazy<Mutex<Vec<Expression<f64>>>> = Lazy::new(|| read_dataset(DATASET_PATH).into());

static DATASET_STRINGIFIED: Lazy<Mutex<Vec<String>>> = Lazy::new(|| {
    let dataset = read_dataset(DATASET_PATH);
    dataset
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .into()
});

fn read_dataset<P: AsRef<Path>>(path: P) -> Vec<Expression<f64>> {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    let buffer = lz4_flex::decompress_size_prepended(&buffer).unwrap();
    bincode::deserialize(&buffer).unwrap()
}

// enum CacheFriendly<'a> {
//     Value(f64),
//     Product {
//         left: BBox<'a, Self>,
//         right: BBox<'a, Self>,
//     }
// }

// enum NonSoCacheFriendly {
//     Value(f64),
//     Product {
//         left: Box<Self>,
//         right: Box<Self>,
//     }
// }

fn evaluate() {
    let dataset = DATASET.lock().unwrap();
    for expression in dataset.iter() {
        iai::black_box(expression.evaluate());
    }
}

fn parse() {
    let dataset = DATASET_STRINGIFIED.lock().unwrap();

    let parser = TopLevelExpressionParser::new();
    let mut context = Context::default();

    for expression in dataset.iter() {
        context.clear();
        iai::black_box(parser.parse(&mut context, expression).unwrap());
    }
}

iai::main!(evaluate, parse);
