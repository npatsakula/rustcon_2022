use crate::{function::Function, prelude::*};

use recursion::{
    map_layer::MapLayer,
    recursive_tree::{ArenaIndex, RecursiveTree},
    Collapse, Expand,
};

pub enum Layer<'input, V, N> {
    Value(V),

    Op {
        left: N,
        op: Operation,
        right: N,
    },

    FnCall {
        function: Function<'input>,
        arguments: Vec<N>,
    },
}

impl<'l, 'input, A: Copy, B: 'l, V: Copy> MapLayer<B> for &'l Layer<'input, V, A> {
    type To = Layer<'input, V, B>;
    type Unwrapped = A;

    #[inline(always)]
    fn map_layer<F: FnMut(Self::Unwrapped) -> B>(self, mut f: F) -> Self::To {
        match self {
            Layer::Value(v) => Self::To::Value(*v),
            Layer::Op { left, op, right } => Self::To::Op {
                left: f(*left),
                op: *op,
                right: f(*right),
            },
            Layer::FnCall {
                function,
                arguments,
            } => Self::To::FnCall {
                function: function.clone(),
                arguments: arguments.iter().copied().map(f).collect(),
            },
        }
    }
}

impl<'input, A, B, V> MapLayer<B> for Layer<'input, V, A> {
    type To = Layer<'input, V, B>;
    type Unwrapped = A;

    #[inline(always)]
    fn map_layer<F: FnMut(Self::Unwrapped) -> B>(self, mut f: F) -> Self::To {
        match self {
            Layer::Value(v) => Self::To::Value(v),
            Layer::Op { left, op, right } => Self::To::Op {
                left: f(left),
                op,
                right: f(right),
            },
            Layer::FnCall {
                function,
                arguments,
            } => Self::To::FnCall {
                function,
                arguments: arguments.into_iter().map(f).collect(),
            },
        }
    }
}

pub type TopBlockExpression<'input> = BlockExpression<'input, f64>;

pub struct BlockExpression<'input, V> {
    inner: RecursiveTree<Layer<'input, V, ArenaIndex>, ArenaIndex>,
}

impl<'input, V: Copy> BlockExpression<'input, V> {
    #[inline]
    pub fn new(source: Expression<'input, V>) -> Self {
        Self {
            inner: RecursiveTree::expand_layers(source, |layer| match layer {
                Expression::Value(v) => Layer::Value(v),
                Expression::Op { left, op, right } => Layer::Op {
                    left: *left,
                    op,
                    right: *right,
                },
                Expression::FnCall {
                    function,
                    arguments,
                } => Layer::FnCall {
                    function,
                    arguments,
                },
            }),
        }
    }
}

impl<'input> TopBlockExpression<'input> {
    pub fn evaluate(&self) -> f64 {
        self.inner.as_ref().collapse_layers(|node: Layer<f64, f64>| match node {
            Layer::Value(v) => v,
            Layer::Op { left, op, right } => op.evaluate(left, right),
            Layer::FnCall {
                function,
                arguments,
            } => function.evaluate(
                &arguments
                    .into_iter()
                    .map(Expression::Value)
                    .collect::<Vec<_>>(),
            ),
        })
    }
}

#[cfg(test)]
mod block_tests {
    use proptest::{prop_assert_eq, strategy::Strategy};

    use crate::block::TopBlockExpression;

    proptest::proptest! {
        #[test]
        fn expression_equality(source in crate::properties::top_level_expression().prop_filter("must be normal", |e| e.evaluate().is_normal())) {
            let evaluated = source.evaluate();

            let block = TopBlockExpression::new(source);
            prop_assert_eq!(evaluated, block.evaluate());
        }
    }
}
