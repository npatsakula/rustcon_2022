use std::{
    cmp::Ordering,
    io::{Read, Write},
    path::Path,
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use evac::{grammar::TopLevelExpressionParser, Expression, function::Context};
use pprof::criterion::{PProfProfiler, Output};
use proptest::{
    strategy::{Strategy, ValueTree},
    test_runner::TestRunner,
};

#[global_allocator]
static ALLOC: mimalloc_rust::GlobalMiMalloc = mimalloc_rust::GlobalMiMalloc;

#[allow(dead_code)]
fn gen_expression() -> impl Iterator<Item = Expression<f64>> {
    let mut runner = TestRunner::default();
    let mut generator = evac::properties::expression()
        .new_tree(&mut runner)
        .unwrap();

    std::iter::from_fn(move || {
        let expr: Expression<_> = generator.current();
        let leafs_count = expr.leafs_count();

        match leafs_count.cmp(&30) {
            Ordering::Less => generator.simplify(),
            Ordering::Greater => generator.complicate(),
            Ordering::Equal => true,
        };

        Some(expr)
    })
}

fn read_dataset<P: AsRef<Path>>(path: P) -> Vec<Expression<f64>> {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    let buffer = lz4_flex::decompress_size_prepended(&buffer).unwrap();
    bincode::deserialize(&buffer).unwrap()
}

#[allow(dead_code)]
fn write_dataset<P: AsRef<Path>>(path: P, dataset: Vec<Expression<f64>>) {
    let file = std::fs::File::create(path).unwrap();
    let mut file = std::io::BufWriter::new(file);
    let dataset = bincode::serialize(&dataset).unwrap();
    let dataset = lz4_flex::compress_prepend_size(&dataset);
    file.write_all(&dataset).unwrap();
}

fn evaluate(c: &mut Criterion) {
    const BATCH_SIZE: usize = 500_000;
    let dataset_path = format!("./data/expressions_{BATCH_SIZE}.bin.lz4");

    let dataset = read_dataset(&dataset_path);
    let dataset_stringified: Vec<_> = dataset.iter().map(|e| e.to_string()).collect();
    let parser = TopLevelExpressionParser::new();
    let mut context = Context::default();

    assert_eq!(dataset.len(), BATCH_SIZE);

    c.bench_with_input(
        BenchmarkId::new("parse", BATCH_SIZE),
        &dataset_stringified,
        |b, input| {
            context.clear();
            b.iter(|| {
                input
                    .iter()
                    .map(|q| parser.parse(&mut context, q).unwrap())
                    .collect::<Vec<_>>()
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("interpretation", BATCH_SIZE),
        &dataset,
        |b, input| b.iter(|| input.iter().map(Expression::evaluate).sum::<f64>()),
    );
}

criterion_group!(
    name = evac; 
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = evaluate
);

criterion_main!(evac);
