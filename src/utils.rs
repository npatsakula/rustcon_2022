use std::{
    io::{Read, Write},
    ops::RangeInclusive,
    path::Path,
};

use proptest::{strategy::Strategy, strategy::ValueTree, test_runner::TestRunner};
use rand::Rng;

use crate::{
    properties::{functions, top_level_expression},
    Expression, Operation, function::Function,
};

// Генерируем одно корректное выражение:
pub fn gen_correct_expression() -> Expression<'static, 'static, f64> {
    let mut runner = TestRunner::default();
    crate::properties::top_level_expression()
        .new_tree(&mut runner)
        .unwrap()
        .current()
}

pub fn generate_correct_call() -> Expression<'static, 'static, f64> {
    let mut runner = TestRunner::default();
    functions()
        .prop_map(|mut fs| {
            fs.sort_by_key(|e| e.expression.leafs_count());
            fs.pop().unwrap()
        })
        .prop_flat_map(|function| {
            proptest::collection::vec(top_level_expression(), function.argument_count()).prop_map(
                move |arguments| Expression::FnCall {
                    function: Function::UserDefined(function.clone()),
                    arguments,
                },
            )
        })
        .new_tree(&mut runner)
        .unwrap()
        .current()
}

// Склеиваем из корректных выражений большее при помощи операции
// с низким приоритетом.
pub fn generate_expression_size(size: usize) -> Expression<'static, 'static, f64> {
    let mut current = 0usize;
    let mut result = Expression::Value(11.0);

    while current < size {
        let generated = gen_correct_expression();
        current += generated.leafs_count();
        result = Expression::op(result, Operation::Sub, generated);
    }

    result
}

// Генерируем вектор из выражений, размер каждого варьируется в отрезке.
pub fn generate_dataset(
    size: usize,
    range: RangeInclusive<usize>,
) -> Vec<Expression<'static, 'static, f64>> {
    (0..size)
        .map(move |_| rand::thread_rng().gen_range(range.clone()))
        .map(generate_expression_size)
        .collect()
}

// При помощи частичного применения делаем генератор по-умолчанию.
pub fn generate_default_dataset() -> Vec<Expression<'static, 'static, f64>> {
    generate_dataset(3_000, 900..=950)
}

// Функция десериализации тестового набора данных. Данные сжимаются
// при помощи нативного компрессора LZ4, за счёт чего используется
// меньше места на диске.
pub fn read_dataset<P: AsRef<Path>>(path: P) -> Vec<u8> {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    lz4_flex::decompress_size_prepended(&buffer).unwrap()
}

pub fn deserialize_dataset(raw: &[u8]) -> Vec<Expression<f64>> {
    bincode::deserialize(raw).unwrap()
}

// Функция сериализации тестового набора данных.
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
    let raw = read_dataset(DEFAULT_DATASET);
    let deserialized = deserialize_dataset(&raw);
    assert_eq!(dataset, deserialized);
}
