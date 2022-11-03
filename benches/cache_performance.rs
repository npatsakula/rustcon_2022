use std::path::Path;

use criterion::{
    black_box, criterion_group, criterion_main, profiler::Profiler, BenchmarkId, Criterion,
};
use evac::{properties::generate_tree, Expression, Operation};
use recursion::{
    map_layer::MapLayer,
    recursive_tree::{ArenaIndex, RecursiveTree},
    Collapse, Expand,
};

// #[global_allocator]
// static ALLOC: mimalloc_rust::GlobalMiMalloc = mimalloc_rust::GlobalMiMalloc;

#[derive(Clone)]
enum CacheFriendlyLayer<V, N> {
    Value(V),
    Product { left: N, right: N },
    Division { left: N, right: N },
    Sum { left: N, right: N },
    Subtraction { left: N, right: N },
}

type CacheFriendly<V> = RecursiveTree<CacheFriendlyLayer<V, ArenaIndex>, ArenaIndex>;
type CacheFriendlyFloat = CacheFriendly<f64>;

impl<A, B, V> MapLayer<B> for CacheFriendlyLayer<V, A> {
    type To = CacheFriendlyLayer<V, B>;
    type Unwrapped = A;

    #[inline(always)]
    fn map_layer<F: FnMut(Self::Unwrapped) -> B>(self, mut f: F) -> Self::To {
        match self {
            Self::Value(v) => Self::To::Value(v),
            Self::Product { left, right } => Self::To::Product {
                left: f(left),
                right: f(right),
            },
            Self::Division { left, right } => Self::To::Division {
                left: f(left),
                right: f(right),
            },
            Self::Sum { left, right } => Self::To::Sum {
                left: f(left),
                right: f(right),
            },
            Self::Subtraction { left, right } => Self::To::Subtraction {
                left: f(left),
                right: f(right),
            },
        }
    }
}

impl<'a, A: Copy, B: 'a, V: Copy> MapLayer<B> for &'a CacheFriendlyLayer<V, A> {
    type To = CacheFriendlyLayer<V, B>;
    type Unwrapped = A;

    #[inline(always)]
    fn map_layer<F: FnMut(Self::Unwrapped) -> B>(self, mut f: F) -> Self::To {
        match self {
            CacheFriendlyLayer::Value(v) => Self::To::Value(*v),
            CacheFriendlyLayer::Product { left, right } => Self::To::Product {
                left: f(*left),
                right: f(*right),
            },
            CacheFriendlyLayer::Division { left, right } => Self::To::Division {
                left: f(*left),
                right: f(*right),
            },
            CacheFriendlyLayer::Sum { left, right } => Self::To::Sum {
                left: f(*left),
                right: f(*right),
            },
            CacheFriendlyLayer::Subtraction { left, right } => Self::To::Subtraction {
                left: f(*left),
                right: f(*right),
            },
        }
    }
}

fn from_boxed(source: &Expression<f64>) -> CacheFriendlyFloat {
    use Operation::*;
    CacheFriendlyFloat::expand_layers(source, |node| match node {
        Expression::Value(v) => CacheFriendlyLayer::Value(*v),
        Expression::Op {
            left,
            right,
            op: Product,
        } => CacheFriendlyLayer::Product {
            left: left.as_ref(),
            right: right.as_ref(),
        },
        Expression::Op {
            left,
            right,
            op: Minus,
        } => CacheFriendlyLayer::Subtraction {
            left: left.as_ref(),
            right: right.as_ref(),
        },
        Expression::Op {
            left,
            right,
            op: Plus,
        } => CacheFriendlyLayer::Sum {
            left: left.as_ref(),
            right: right.as_ref(),
        },
        Expression::Op {
            left,
            right,
            op: Division,
        } => CacheFriendlyLayer::Division {
            left: left.as_ref(),
            right: right.as_ref(),
        },
        Expression::FnCall { .. } => unreachable!(),
    })
}

fn performance(c: &mut Criterion) {
    const NODES: usize = 100_000;
    let not_local = black_box(generate_tree(NODES));
    let local = from_boxed(black_box(&not_local));
    let count = not_local.leafs_count();

    let mut group = c.benchmark_group("cache perfomance");

    group.bench_with_input(BenchmarkId::new("cache friendly", count), &local, |b, i| {
        b.iter(|| {
            i.as_ref().collapse_layers(|n| match n {
                CacheFriendlyLayer::Value(v) => v,
                CacheFriendlyLayer::Product { left, right } => left * right,
                CacheFriendlyLayer::Division { left, right } => left / right,
                CacheFriendlyLayer::Sum { left, right } => left + right,
                CacheFriendlyLayer::Subtraction { left, right } => left - right,
            })
        })
    });

    group.bench_with_input(
        BenchmarkId::new("not cache friendly", count),
        &not_local,
        |b, i| b.iter(|| i.evaluate()),
    );

    group.finish();
}

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

criterion_group!(
    name = cache_performance;
    config = Criterion::default().with_profiler(GProfiler);
    targets = performance
);

criterion_main!(cache_performance);
