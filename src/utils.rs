use std::{
    io::{Read, Write},
    ops::RangeInclusive,
    path::Path,
};

use proptest::{strategy::Strategy, strategy::ValueTree, test_runner::TestRunner};
use rand::Rng;

use crate::{Expression, Operation};

pub fn gen_correct_expression() -> Expression<'static, f64> {
    let mut runner = TestRunner::default();
    crate::properties::expression()
        .new_tree(&mut runner)
        .unwrap()
        .current()
}

pub fn generate_expression_size(size: usize) -> Expression<'static, f64> {
    let mut current = 0usize;
    let mut result = Expression::Value(11.0);

    while current < size {
        let generated = gen_correct_expression();
        current += generated.leafs_count();
        result = Expression::op(result, Operation::Minus, generated);
    }

    result
}

pub fn generate_dataset(
    size: usize,
    range: RangeInclusive<usize>,
) -> Vec<Expression<'static, f64>> {
    (0..size)
        .map(move |_| rand::thread_rng().gen_range(range.clone()))
        .map(generate_expression_size)
        .collect()
}

pub fn generate_default_dataset() -> Vec<Expression<'static, f64>> {
    generate_dataset(3_000, 900..=950)
}

pub fn read_dataset<P: AsRef<Path>>(path: P) -> Vec<Expression<'static, f64>> {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    let buffer = lz4_flex::decompress_size_prepended(&buffer).unwrap();
    bincode::deserialize(&buffer).unwrap()
}

pub fn write_dataset<P: AsRef<Path>>(path: P, dataset: Vec<Expression<f64>>) {
    let file = std::fs::File::create(path).unwrap();
    let mut file = std::io::BufWriter::new(file);
    let dataset = bincode::serialize(&dataset).unwrap();
    let dataset = lz4_flex::compress_prepend_size(&dataset);
    file.write_all(&dataset).unwrap();
}

pub const DEFAULT_DATASET: &str = "./data/expression_3k.bin.lz4";

#[test]
#[ignore = "For dataset generation only."]
fn generate_dataset_test() {
    let dataset = generate_default_dataset();
    write_dataset(DEFAULT_DATASET, dataset.clone());
    let deserialized = read_dataset(DEFAULT_DATASET);
    assert_eq!(dataset, deserialized);
}
