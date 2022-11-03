use std::{collections::HashSet, fmt::Debug, sync::Arc};

use crate::{function::Function, prelude::*};
use proptest::{prelude::*, sample::select};
use Operation::*;

pub fn value() -> impl Strategy<Value = f64> + Clone {
    any::<f64>().prop_filter("Values must be comparable.", |x| x.is_normal())
}

pub fn function_value(variables: Vec<String>) -> impl Strategy<Value = FunctionValue> + Clone {
    prop_oneof![
        // Число должно генерироваться в пять раз чаще, чем переменная.
        5 => value().prop_map(FunctionValue::Float),
        1 => select(variables).prop_map(FunctionValue::Variable)
    ]
}

fn term<S: Strategy<Value = V> + Clone + 'static, V: Clone + Debug + 'static>(
    value_strategy: S,
    user_defined_functions: Vec<Arc<UserDefinedFunction>>,
) -> impl Strategy<Value = Expression<V>> {
    prop_oneof![
        // Числово должно генерироваться в десять раз чаще, чем вызов функции.
        10 => value_strategy.clone().prop_map(Expression::Value),
        1 => function_call(value_strategy, user_defined_functions),
    ]
}

fn user_defined_function(
    functions: Vec<Arc<UserDefinedFunction>>,
) -> impl Strategy<Value = UserDefinedFunction> {
    (
        // Имя функции имеет длину от восьми до шестнадцати символов и состоит из
        // символов класса `alpha`.
        r"\p{alpha}{8,16}",
        // Функция имеет от двух до пяти аргументов. Каждый аргумент имеет имя
        // состоящие из имволов класса `alpha` и содержащие от трёх до семи
        // символов.
        //
        // Так как имена аргументов не должны совпадать, генерировать мы будем `HashSet`.
        proptest::collection::hash_set(r"\p{alpha}{3,7}", 2..5),
    )
        .prop_flat_map(move |(name, args)| {
            let args = args.into_iter().collect::<Vec<_>>();
            expression(function_value(args.clone()), functions.clone()).prop_map(
                move |expression| UserDefinedFunction {
                    name: name.clone(),
                    args: args.clone(),
                    expression,
                },
            )
        })
}

fn function_call<S: Strategy<Value = V> + Clone + 'static, V: Clone + Debug + 'static>(
    value_strategy: S,
    user_defined_functions: Vec<Arc<UserDefinedFunction>>,
) -> impl Strategy<Value = Expression<V>> {
    // Если пользовательских функций нет, то создаём только встроенные.
    // Если на пустом векторе позвать `select`, получим ошибку генерации данных.
    if user_defined_functions.is_empty() {
        any::<BuiltinFunction>()
            .prop_map(Into::<Function>::into)
            // Боксирование происходит, чтобы не разошлись типы выражений,
            // возвращаемых if-else.
            .boxed()
    } else {
        prop_oneof![
            // Функции, которые определены пользователем, используются в десять раз реже
            // встроенных функций.
            1 => select(user_defined_functions.clone()).prop_map(Function::UserDefined),
            10 => any::<BuiltinFunction>().prop_map(Into::<Function>::into),
        ]
        .boxed()
    }
    .prop_flat_map(move |function| {
        prop::collection::vec(
            // Генерируем аргументы функции:
            expression(value_strategy.clone(), user_defined_functions.clone()),
            function.arguments_count(),
        )
        .prop_map(move |arguments| Expression::FnCall {
            function: function.clone(),
            arguments,
        })
    })
}

fn factor<S: Strategy<Value = V> + Clone + 'static, V: Clone + Debug + 'static>(
    strategy: S,
    user_defined_functions: Vec<Arc<UserDefinedFunction>>,
) -> impl Strategy<Value = Expression<V>> {
    // Так как `factor` определяется рекурсивно, мы хотим ограничить глубину рекурсии, чтобы не
    // пробить стэк при генерации данных. Для этого `proptest` предоставляет метод `prop_recursive`,
    // где первым аргументом идёт максимальная глубина рекурсии, вторым желаемое количество элементов,
    // а третьим желаемая глубина рекурсии. Четвёртым же элементом идёт функция, которая единственным
    // элементом принимает стратегию получения `factor`, которую мы можем использовать для рекурсивного
    // определения.
    term(strategy.clone(), user_defined_functions.clone()).prop_recursive(4, 16, 4, move |inner| {
        (
            inner,
            prop_oneof![Just(Mul), Just(Div)],
            term(strategy.clone(), user_defined_functions.clone()),
        )
            .prop_map(|(left, op, right)| Expression::op(left, op, right))
    })
}

fn expression<S: Strategy<Value = V> + Clone + 'static, V: Clone + Debug + 'static>(
    value_strategy: S,
    user_defined_functions: Vec<Arc<UserDefinedFunction>>,
) -> impl Strategy<Value = Expression<V>> {
    factor(value_strategy.clone(), user_defined_functions.clone())
        .prop_recursive(4, 16, 4, move |inner| {
            (
                inner,
                prop_oneof![Just(Sum), Just(Sub)],
                factor(value_strategy.clone(), user_defined_functions.clone()),
            )
                .prop_map(|(expr, op, factor)| Expression::op(expr, op, factor))
        })
        .boxed()
}

// Набор функций, которые будут использоваться в выражении, будет определён рекурсивно.
// Делается это потому, что одни функции могу ссылаться на другие (определённые ранее).
fn functions() -> impl Strategy<Value = Vec<Arc<UserDefinedFunction>>> {
    // В качестве вырожденного случая будет использоваться функция, которая не ссылается ни
    // на какие другие функции.
    user_defined_function(Vec::new())
        .prop_map(|f| vec![Arc::new(f)])
        // Мы будем рекурсивно расширять перечень доступных функций следующим образом:
        .prop_recursive(16, 128, 16, |inner| {
            inner.prop_flat_map(|fs| {
                user_defined_function(fs.clone()).prop_map(move |f| {
                    let mut fs = fs.clone();
                    fs.push(f.into());
                    fs
                })
            })
        })
        // Имена функций обязаны быть уникальными, здесь мы делаем соответствующую проверку:
        .prop_filter("functions must be unique", |functions| {
            let names: HashSet<&str> = functions.iter().map(|f| f.name.as_str()).collect();
            names.len() == functions.len()
        })
}

pub fn top_level_expression() -> impl Strategy<Value = Expression<f64>> {
    // Сначала мы генерируем функции, которые планируем использовать, после чего генерируем
    // выражение:
    functions().prop_flat_map(|user_defined_functions| expression(value(), user_defined_functions))
}
