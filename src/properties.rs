use crate::prelude::*;
use proptest::{prelude::*, test_runner::TestRunner, strategy::ValueTree};
use Operation::*;

pub fn value_raw() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("Values must be comparable.", |x| x.is_normal())
}

fn term() -> impl Strategy<Value = Expression<'static, f64>> {
    value_raw().prop_map(Expression::Value)
}

// fn function_call() -> impl Strategy<Value = Expression<'static, f64>> {
//     let name = prop_oneof![Just("sin"), Just("cos")];
//     (name, term()).prop_map(|(name, arg)| Expression::FnCall {
//         arguments: vec![arg],
//         function: (if name == "sin" {
//             BuiltinFunction::Sin
//         } else {
//             BuiltinFunction::Cos
//         })
//         .into(),
//     })
// }

fn factor() -> impl Strategy<Value = Expression<'static, f64>> {
    term()
        .prop_recursive(128, 1024, 100, |inner| {
            (inner, prop_oneof![Just(Product), Just(Division)], term())
                .prop_map(|(left, op, right)| Expression::op(left, op, right))
        })
}

pub fn expression() -> impl Strategy<Value = Expression<'static, f64>> {
    factor()
        .prop_recursive(128, 1024, 100, |inner| {
            (inner, prop_oneof![Just(Plus), Just(Minus)], factor())
                .prop_map(|(expr, op, factor)| Expression::op(expr, op, factor))
        })
}

pub fn generate_tree(nodes: usize) -> Expression<'static, f64> {
    let config = proptest::test_runner::Config {
        max_local_rejects: u32::MAX,
        ..Default::default()
    };
    let mut runner = TestRunner::new(config);
    let generator = expression()
        .no_shrink()
        .new_tree(&mut runner)
        .unwrap();
    
    let generator = || generator.current();
    let mut count = 0;
    let mut result = Expression::Value(16.0);

    while count < nodes {
        result = Expression::op(generator(), Plus, result);
        count = result.leafs_count();
    }

    result
}
