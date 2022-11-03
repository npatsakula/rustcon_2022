use std::path::Path;

use criterion::{
    black_box, criterion_group, criterion_main, profiler::Profiler, BenchmarkId, Criterion,
};
use evac::{
    function::Context,
    grammar::TopLevelExpressionParser,
    utils::{read_dataset, DEFAULT_DATASET},
    Expression,
};
use pprof::criterion::{Output, PProfProfiler};

fn benchmark(c: &mut Criterion) {
    // Десериализуем датасет, который сгенерировали ранее.
    let dataset = read_dataset(DEFAULT_DATASET);
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
                    black_box(parser.parse(&mut Context::default(), q).unwrap());
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
