use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
    module::Module,
    passes::PassManager,
    types::FloatType,
    values::{BasicMetadataValueEnum, FloatValue, FunctionValue},
    OptimizationLevel,
};
use itertools::Itertools;

use crate::function::FunctionValue as FValue;
use crate::{Expression, Operation};

use super::{FunctionTrait, UserDefinedFunction};
use crate::error::*;

#[derive(Debug)]
pub struct Codegen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    pass_manager: PassManager<FunctionValue<'ctx>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvacJitFunction<'i, 'ctx> {
    name: &'i str,
    inner: JitFunctionInner<'ctx>,
}

impl<'i, 'ctx> FunctionTrait for EvacJitFunction<'i, 'ctx> {
    fn name(&self) -> &str {
        self.name
    }

    fn arguments_count(&self) -> usize {
        match &self.inner {
            JitFunctionInner::Constant(_) => 0,
            JitFunctionInner::One(_) => 1,
            JitFunctionInner::Two(_) => 2,
            JitFunctionInner::Three(_) => 3,
            JitFunctionInner::Four(_) => 4,
            JitFunctionInner::Five(_) => 5,
        }
    }

    fn evaluate(&self, arguments: &[Expression<f64>]) -> Result<f64, crate::prelude::Error> {
        snafu::ensure!(
            arguments.len() == self.arguments_count(),
            WrongArgumentsAmountSnafu {
                name: self.name.to_string(),
                expected: self.arguments_count(),
                parsed: arguments.len(),
            }
        );

        let arguments = arguments.iter().map(Expression::evaluate).collect_vec();
        Ok(self.inner.evaluate(&arguments))
    }
}

#[derive(Debug, Clone)]
pub enum JitFunctionInner<'ctx> {
    Constant(JitFunction<'ctx, unsafe extern "C" fn() -> f64>),
    One(JitFunction<'ctx, unsafe extern "C" fn(f64) -> f64>),
    Two(JitFunction<'ctx, unsafe extern "C" fn(f64, f64) -> f64>),
    Three(JitFunction<'ctx, unsafe extern "C" fn(f64, f64, f64) -> f64>),
    Four(JitFunction<'ctx, unsafe extern "C" fn(f64, f64, f64, f64) -> f64>),
    Five(JitFunction<'ctx, unsafe extern "C" fn(f64, f64, f64, f64, f64) -> f64>),
}

impl PartialEq for JitFunctionInner<'_> {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<'ctx> JitFunctionInner<'ctx> {
    pub fn evaluate(&self, args: &[f64]) -> f64 {
        unsafe {
            match self {
                Self::Constant(f) => f.call(),
                Self::One(f) => f.call(args[0]),
                Self::Two(f) => f.call(args[0], args[1]),
                Self::Three(f) => f.call(args[0], args[1], args[2]),
                Self::Four(f) => f.call(args[0], args[1], args[2], args[3]),
                Self::Five(f) => f.call(args[0], args[1], args[2], args[3], args[4]),
            }
        }
    }
}

impl<'ctx> Codegen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("jit");
        let builder = context.create_builder();
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        let pass = PassManager::create(&module);

        pass.add_instruction_combining_pass();
        pass.add_instruction_simplify_pass();
        pass.add_new_gvn_pass();
        pass.add_cfg_simplification_pass();
        pass.add_reassociate_pass();
        pass.add_function_inlining_pass();

        pass.initialize();

        Self {
            context,
            module,
            builder,
            execution_engine,
            pass_manager: pass,
        }
    }

    pub fn operation(
        &self,
        operation: Operation,
        left: FloatValue<'ctx>,
        right: FloatValue<'ctx>,
    ) -> FloatValue<'ctx> {
        match operation {
            Operation::Sum => self.builder.build_float_add(left, right, "sum"),
            Operation::Mul => self.builder.build_float_mul(left, right, "mul"),
            Operation::Div => self.builder.build_float_div(left, right, "div"),
            Operation::Sub => self.builder.build_float_sub(left, right, "sub"),
        }
    }

    pub fn function_expression(
        &self,
        expression: &Expression<FValue>,
        function: &FunctionValue<'ctx>,
        args: &[&str],
    ) -> FloatValue<'ctx> {
        let expr = |expr| self.function_expression(expr, function, args);
        match expression {
            Expression::Value(FValue::Float(v)) => self.context.f64_type().const_float(*v),
            Expression::Value(FValue::Variable(v)) => {
                let index = args.iter().position(|a| a == v).unwrap();
                function
                    .get_nth_param(index as u32)
                    .unwrap()
                    .into_float_value()
            }
            Expression::Op { left, op, right } => {
                let left = expr(left.as_ref());
                let right = expr(right.as_ref());
                self.operation(*op, left, right)
            }
            Expression::FnCall {
                function,
                arguments,
            } => {
                let function = self.module.get_function(function.name()).unwrap();

                let args = arguments
                    .iter()
                    .map(expr)
                    .map(Into::<BasicMetadataValueEnum>::into)
                    .collect::<Vec<_>>();

                self.builder
                    .build_call(function, &args, "call")
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value()
            }
        }
    }

    pub fn user_defined_function(
        &self,
        UserDefinedFunction {
            name,
            args,
            expression,
        }: &UserDefinedFunction,
    ) -> FunctionValue {
        let f64_type = self.context.f64_type();
        let function_type = f64_type.fn_type(&vec![f64_type.into(); args.len()], false);
        let function = self.module.add_function(name, function_type, None);

        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let res = self.function_expression(expression, &function, args);
        self.builder.build_return(Some(&res));


        let result = self.execution_engine.get_function_value(name).unwrap();

        if !result.verify(true) {
            panic!("Failed to build function.");
        } else {
            result
        }
    }

    pub fn pi(&self) -> Option<FunctionValue<'ctx>> {
        let f64_type: FloatType = self.context.f64_type();
        let function_type = f64_type.fn_type(&[], false);

        let function = self.module.add_function("pi", function_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let result = f64_type.const_float(std::f64::consts::PI);
        self.builder.build_return(Some(&result));

        self.pass_manager.run_on(&function);

        let function = self.execution_engine.get_function_value("pi").ok()?;
        Some(function)
    }
}

#[test]
fn simple_pi() {
    let context = Context::create();
    let codegen = Codegen::new(&context);
    let function = codegen.pi().unwrap();
    assert!(function.verify(true));
    let caller: JitFunctionInner = unsafe {
        codegen
            .execution_engine
            .get_function("pi")
            .map(JitFunctionInner::Constant)
            .unwrap()
    };

    assert_eq!(caller.evaluate(&[]), std::f64::consts::PI);
}

#[test]
fn simple_fn() {
    let context = Context::create();
    let codegen = Codegen::new(&context);

    let expression = Expression::op(
        FValue::Float(2.0).into(),
        Operation::Sum,
        FValue::Variable("foo").into(),
    );

    let function = UserDefinedFunction {
        name: "plus_two",
        args: vec!["foo"],
        expression,
    };

    let _function = codegen.user_defined_function(&function);

    let result = unsafe { codegen.execution_engine.get_function("plus_two") }
        .map(JitFunctionInner::One)
        .unwrap()
        .evaluate(&[2.0]);

    assert_eq!(result, 4.0)
}

proptest::proptest! {
    // #[test]
    fn function(expression in crate::properties::top_level_expression()) {
        let jit_context = Context::create();
        let codegen = Codegen::new(&jit_context);

        let context = crate::prelude::Context::from_expression(&expression);
        let functions = context.sort_calls();

        for function in functions {
            codegen.user_defined_function(function.as_ref());
        }
    }
}
