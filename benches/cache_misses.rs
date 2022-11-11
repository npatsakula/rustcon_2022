use evac::{block::Layer, Expression, Operation};
use iai::black_box;

#[repr(C)]
struct BoxedSum {
    left: Box<i32>,
    right: Box<i32>,
}

impl BoxedSum {
    fn new(left: i32, right: i32) -> Self {
        Self {
            left: left.into(),
            right: right.into(),
        }
    }

    fn evaluate(&self) -> i32 {
        *self.left + *self.right
    }
}

#[repr(C)]
struct LocalSum {
    offset_1: usize,
    left: i32,
    offset_2: usize,
    right: i32,
}

impl LocalSum {
    fn new(left: i32, right: i32) -> Self {
        Self {
            offset_1: 2,
            left,
            offset_2: 4,
            right,
        }
    }

    fn evaluate(&self) -> i32 {
        self.left + self.right
    }
}

fn not_cache_local() {
    let sum = BoxedSum::new(black_box(3), black_box(4));
    black_box(sum.evaluate());
    black_box(sum);
}

fn cache_local() {
    let sum = LocalSum::new(black_box(3), black_box(4));
    black_box(sum.evaluate());
    black_box(sum);
}

fn real_expression() {
    black_box(
        Expression::op(
            black_box(Expression::Value(3.0)),
            black_box(Operation::Plus),
            black_box(Expression::Value(4.0)),
        )
        .evaluate(),
    );
}

fn real_block() {
    let data = vec![
        Layer::<'static, f64, ()>::Value(0.0), // Offset.
        black_box(Layer::Value(3.0)),
        Layer::Value(0.0), // Offset.
        black_box(Layer::Value(4.0)),
    ];
    let layer: Layer<'static, (), usize> = Layer::Op {
        left: black_box(1),
        op: Operation::Plus,
        right: black_box(3),
    };

    if let Layer::Op {
        left,
        op: Operation::Plus,
        right,
    } = layer
    {
        let left = if let Layer::Value(value) = unsafe { data.get_unchecked(left) } {
            value
        } else {
            return;
        };

        let right = if let Layer::Value(value) = unsafe { data.get_unchecked(right) } {
            value
        } else {
            return;
        };

        black_box(left + right);
    }
}

iai::main!(not_cache_local, cache_local, real_expression, real_block);
