use std::path::Path;

use criterion::{
    black_box, criterion_group, criterion_main, profiler::Profiler, BenchmarkId, Criterion,
};
use evac::{
    block::TopBlockExpression,
    function::{Context, jit::Codegen},
    grammar::TopLevelExpressionParser,
    lexer::EvacLexer,
    utils::{deserialize_dataset, read_dataset, DEFAULT_DATASET, generate_correct_call},
    Expression,
};
use itertools::Itertools;
use pprof::criterion::{Output, PProfProfiler};

fn benchmark(c: &mut Criterion) {
    // Десериализуем датасет, который сгенерировали ранее.
    let raw = read_dataset(DEFAULT_DATASET);
    let dataset = deserialize_dataset(&raw);
    let dataset_size = dataset.len();

    let dataset_stringified: Vec<_> = dataset
        .iter()
        .map(|e| {
            // let context = Context::from_expression(e);
            // // Мы не готовы "проворачивать фарш обратно" (превраащать контекст
            // // в строку), т.к. это потребует разрешения зависимостей между
            // // функциями, что совершенно выходит за рамки текущего доклада.
            // (context, e.to_string())
            let context = Context::from_expression(e).to_string();
            format!("{context}\n{e}")
        })
        .collect();

    // Мы один раз инициализиуем парсер (структура хранит статические таблицы
    // переходов), чтобы использовать его во всех тестах:
    let parser = TopLevelExpressionParser::new();

    c.bench_with_input(
        BenchmarkId::new("parse", dataset_size),
        &dataset_stringified,
        |b, input| {
            b.iter(|| {
                input.iter().for_each(|q| {
                    // Используем `black_box`, чтобы гарантировать, что компилятор не
                    // удалит результаты этого вычисления в ходе оптимизации.
                    black_box(
                        parser
                            .parse(&mut Context::default(), EvacLexer::new(q))
                            .unwrap(),
                    );
                });
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("not cache local", dataset_size),
        &dataset,
        |b, input| {
            b.iter(|| {
                input.iter().map(Expression::evaluate).for_each(|r| {
                    black_box(r);
                })
            })
        },
    );

    let block_dataset = dataset
        .iter()
        .cloned()
        .map(TopBlockExpression::new)
        .collect::<Vec<_>>();

    c.bench_with_input(
        BenchmarkId::new("cache local", dataset_size),
        &block_dataset,
        |b, input| b.iter(|| input.iter().map(TopBlockExpression::evaluate).sum::<f64>()),
    );

    let calls = (0..1000).map(|_| generate_correct_call()).collect_vec();

    let jit_context = inkwell::context::Context::create();
    let dataset = calls.clone().into_iter().map(|expr| {
        let codegen = Codegen::new(&jit_context);
        for f in  Context::from_expression(&expr).sort_calls() {
            codegen.user_defined_function(f.as_ref()).unwrap();
        }

        expr.jitify_inner(&codegen)
    }).collect_vec();

    c.bench_with_input(
        BenchmarkId::new("compiled calls", dataset_size),
        &dataset,
        |b, input| b.iter(|| input.iter().map(Expression::evaluate).sum::<f64>()),
    );

    c.bench_with_input(
        BenchmarkId::new("calls", dataset_size),
        &calls,
        |b, input| b.iter(|| input.iter().map(Expression::evaluate).sum::<f64>()),
    );
}

criterion_group!(
    name = evac;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    // config = Criterion::default().with_profiler(GProfiler);
    targets = benchmark
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
