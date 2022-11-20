use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction, UnsafeFunctionPointer},
    intrinsics::Intrinsic,
    module::Module,
    passes::PassManager,
    types::FloatType,
    values::{BasicMetadataValueEnum, FloatValue, FunctionValue},
    OptimizationLevel,
};
use itertools::Itertools;
use snafu::ResultExt;

use crate::function::FunctionValue as FValue;
use crate::{Expression, Operation};

use super::{BuiltinFunction, Function, FunctionTrait, UserDefinedFunction};
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

impl<'i, 'ctx> FunctionTrait<'i> for EvacJitFunction<'i, 'ctx> {
    fn name(&self) -> &'i str {
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

        pass.add_verifier_pass();
        pass.add_instruction_combining_pass();
        pass.add_instruction_simplify_pass();
        pass.add_new_gvn_pass();
        pass.add_cfg_simplification_pass();
        pass.add_reassociate_pass();

        pass.initialize();

        let result = Self {
            context,
            module,
            builder,
            execution_engine,
            pass_manager: pass,
        };

        result.pi();
        result.cos();
        result.sin();

        result
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

    fn builtin_name(function: &BuiltinFunction) -> &'static str {
        match function {
            BuiltinFunction::Pi => "pi",
            BuiltinFunction::Cos => "cosine",
            BuiltinFunction::Sin => "sine",
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
                let name = match function {
                    super::Function::Builtin(b) => Self::builtin_name(b),
                    otherwise => otherwise.name(),
                };

                let function = self.module.get_function(name).unwrap();

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

    fn get_function<F: UnsafeFunctionPointer>(&self, name: &str) -> Option<JitFunction<'ctx, F>> {
        unsafe {
            self.execution_engine.remove_module(&self.module).unwrap();
            self.execution_engine.add_module(&self.module).unwrap();

            self.execution_engine
                .get_function(name)
                .context(FunctionLookupSnafu {
                    function: name,
                    module: self.module.to_string(),
                })
                .ok()
        }
    }

    fn wrap_function(&self, name: &str, args_count: usize) -> Option<JitFunctionInner<'ctx>> {
        Some(match args_count {
            0 => JitFunctionInner::Constant(self.get_function(name)?),
            1 => JitFunctionInner::One(self.get_function(name)?),
            2 => JitFunctionInner::Two(self.get_function(name)?),
            3 => JitFunctionInner::Three(self.get_function(name)?),
            4 => JitFunctionInner::Four(self.get_function(name)?),
            5 => JitFunctionInner::Five(self.get_function(name)?),
            _ => todo!(),
        })
    }

    pub fn user_defined_function(
        &self,
        UserDefinedFunction {
            name,
            args,
            expression,
        }: &UserDefinedFunction,
    ) -> Result<JitFunctionInner<'ctx>, Error> {
        if let Some(f) = self.wrap_function(name, args.len()) {
            return Ok(f);
        }

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
            self.pass_manager.run_on(&result);
            Ok(self.wrap_function(name, args.len()).unwrap())
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

    pub fn sin(&self) -> Option<FunctionValue<'ctx>> {
        let f64_type: FloatType = self.context.f64_type();
        let intrinsic = Intrinsic::find("llvm.sin.f64").unwrap();
        let sin = intrinsic
            .get_declaration(&self.module, &[f64_type.into()])
            .unwrap();

        let function_type = f64_type.fn_type(&[f64_type.into()], false);
        let function = self.module.add_function("sine", function_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let result = self
            .builder
            .build_call(
                sin,
                &[function.get_nth_param(0).unwrap().into()],
                "sin_call",
            )
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();

        self.builder.build_return(Some(&result));

        self.pass_manager.run_on(&function);

        let function = self.execution_engine.get_function_value("sine").ok()?;
        Some(function)
    }

    pub fn cos(&self) -> Option<FunctionValue<'ctx>> {
        let f64_type: FloatType = self.context.f64_type();
        let intrinsic = Intrinsic::find("llvm.cos.f64").unwrap();
        let sin = intrinsic
            .get_declaration(&self.module, &[f64_type.into()])
            .unwrap();

        let function_type = f64_type.fn_type(&[f64_type.into()], false);
        let function = self.module.add_function("cosine", function_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let result = self
            .builder
            .build_call(
                sin,
                &[function.get_nth_param(0).unwrap().into()],
                "cos_call",
            )
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();

        self.builder.build_return(Some(&result));

        self.pass_manager.run_on(&function);

        let function = self.execution_engine.get_function_value("cosine").ok()?;
        Some(function)
    }
}

impl<'input, 'ctx: 'input> Codegen<'ctx> {
    pub fn function<'c: 'input>(&self, function: Function<'input, 'ctx>) -> Function<'input, 'ctx> {
        Function::Jited(EvacJitFunction {
            name: function.name(),
            inner: match function {
                Function::Builtin(f) => self
                    .wrap_function(Self::builtin_name(&f), f.arguments_count())
                    .unwrap(),
                Function::UserDefined(ud) => self.user_defined_function(ud.as_ref()).unwrap(),
                Function::Jited(j) => return Function::Jited(j.clone()),
            },
        })
    }
}

#[test]
fn simple_pi() {
    let context = Context::create();
    let codegen = Codegen::new(&context);
    let function = codegen.sin().unwrap();
    // panic!("{}", codegen.module.to_string());
    assert!(function.verify(true));
    let caller: JitFunctionInner = unsafe {
        codegen
            .execution_engine
            .get_function("sine")
            .map(JitFunctionInner::One)
            .unwrap()
    };

    assert_eq!(caller.evaluate(&[1.0]), 1.0f64.sin());
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

    let result = codegen
        .user_defined_function(&function)
        .unwrap()
        .evaluate(&[2.0]);

    assert_eq!(result, 4.0)
}

impl<'input, 'jit: 'input> Expression<'input, 'jit, f64> {
    pub fn jitify_inner(self, codegen: &Codegen<'jit>) -> Self {
        match self {
            v @ Self::Value(_) => v,
            Expression::Op {
                op, left, right, ..
            } => Self::op(left.jitify_inner(codegen), op, right.jitify_inner(codegen)),
            Expression::FnCall {
                function,
                arguments,
            } => {
                let function = codegen.function(function);
                let arguments = arguments
                    .into_iter()
                    .map(|n| n.jitify_inner(codegen))
                    .collect_vec();
                Self::FnCall {
                    function,
                    arguments,
                }
            }
        }
    }
}

#[cfg(test)]
mod jit_test {
    use crate::{
        grammar::TopLevelExpressionParser, lexer::EvacLexer, properties::top_level_expression,
    };

    use super::*;
    use proptest::prelude::*;

    use test_case::test_case;

    #[test_case("
    fn double(foo) = foo * 2.0;
    fn half(foo) = foo / 2.0;

    half(double(1.0))
    " => 1.0; "double and half")]
    #[test_case("
    sin(pi())
    " => std::f64::consts::PI.sin(); "sin and pi")]
    fn jit_me(source: &'static str) -> f64 {
        let mut context = crate::prelude::Context::default();
        let expression = TopLevelExpressionParser::new()
            .parse(&mut context, EvacLexer::new(source))
            .unwrap();

        let jit_context = Context::create();
        let codegen = Codegen::new(&jit_context);
        let jit = expression.jitify_inner(&codegen);

        dbg!(&jit, &context, codegen.module.to_string(),);

        jit.evaluate()
    }

    proptest::proptest! {
        #[test]
        fn generated(expr in top_level_expression().prop_filter("Result must be comparable.", |e| e.evaluate().is_normal())) {
            let jit_context = Context::create();
            let jit_codegen = Codegen::new(&jit_context);
            let jited = expr.clone().jitify_inner(&jit_codegen);

            prop_assert_eq!(
                jited.evaluate(),
                expr.evaluate()
            );
        }
    }
}
