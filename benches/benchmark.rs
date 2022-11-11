use std::path::Path;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, profiler::Profiler};
use evac::{
    block::TopBlockExpression,
    function::Context,
    grammar::TopLevelExpressionParser,
    lexer::EvacLexer,
    utils::{read_dataset, DEFAULT_DATASET},
    Expression,
};
use pprof::criterion::{Output, PProfProfiler};

fn evaluate(c: &mut Criterion) {
    let dataset = read_dataset(DEFAULT_DATASET);
    let dataset_size = dataset.len();

    let dataset_stringified: Vec<_> = dataset.iter().map(|e| e.to_string()).collect();
    let parser = TopLevelExpressionParser::new();

    c.bench_with_input(
        BenchmarkId::new("parse", dataset_size),
        &dataset_stringified,
        |b, input| {
            b.iter(|| {
                input.iter().for_each(|q| {
                    let lexer = EvacLexer::new(q);
                    let mut context = Context::default();
                    black_box(parser.parse(&mut context, lexer).unwrap());
                });
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("not cache local", dataset_size),
        &dataset,
        |b, input| b.iter(|| input.iter().map(Expression::evaluate).sum::<f64>()),
    );

    let dataset = dataset
        .into_iter()
        .map(TopBlockExpression::new)
        .collect::<Vec<_>>();

    c.bench_with_input(
        BenchmarkId::new("cache local", dataset_size),
        &dataset,
        |b, input| b.iter(|| input.iter().map(TopBlockExpression::evaluate).sum::<f64>()),
    );
}

criterion_group!(
    name = evac;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    // config = Criterion::default().with_profiler(GProfiler);
    targets = evaluate
);

criterion_main!(evac);

struct GProfiler;

impl Profiler for GProfiler {
    fn start_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
        std::fs::create_dir_all(benchmark_dir).unwrap();
        let path = benchmark_dir
            .join("gperftools.profile")
            .display()
            .to_string();
        dbg!(&path);
        cpuprofiler::PROFILER.lock().unwrap().start(path).unwrap();
    }

    fn stop_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
        cpuprofiler::PROFILER.lock().unwrap().stop().unwrap();
    }
}
