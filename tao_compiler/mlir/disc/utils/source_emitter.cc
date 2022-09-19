
#include "tensorflow/compiler/mlir/disc/utils/source_emitter.h"

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"

namespace mlir {
namespace disc_ral {

using ValueNameBinding = SourceEmitterCUDA::ValueNameBinding;

template <typename OPType>
std::string CUDAMathFuncName(Type type) {
  assert(false & "unsupported op");
}

template <>
std::string CUDAMathFuncName<lmhlo::AbsOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "__habs";
  } else if (type.isF32() || type.isF64() || type.isInteger(32) ||
             type.isInteger(64)) {
    return "abs";
  } else {
    assert(false & "unsupported type for abs op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::CeilOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hceil";
  } else if (type.isF32() || type.isF64()) {
    return "ceil";
  } else {
    assert(false & "unsupported type for ceil op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::ConvertOp>(Type type) {
  if (type.isBF16()) {
    return "__nv_bfloat162";
  } else if (type.isF16()) {
    return "half";
  } else if (type.isF32()) {
    return "float";
  } else if (type.isF64()) {
    return "double";
  } else if (type.isSignedInteger(32)) {
    return "int";
  } else if (type.isUnsignedInteger(32)) {
    return "(unsigned int)";
  } else if (type.isSignedInteger(64)) {
    return "(long int)";
  } else if (type.isUnsignedInteger(64)) {
    return "(unsigned long int)";
  } else {
    assert(false & "unsupported type for convert op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::CosineOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hcos";
  } else if (type.isF32() || type.isF64()) {
    return "cos";
  } else {
    assert(false & "unsupported type for cosine op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::ExpOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hexp";
  } else if (type.isF32() || type.isF64()) {
    return "exp";
  } else {
    assert(false & "unsupported type for exp op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::FloorOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hfloor";
  } else if (type.isF32() || type.isF64()) {
    return "floor";
  } else {
    assert(false & "unsupported type for floor op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::IsFiniteOp>(Type type) {
  if (type.isF32() || type.isF64()) {
    return "isfinite";
  } else {
    assert(false & "unsupported type for isfinite op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::LogOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hlog";
  } else if (type.isF32() || type.isF64()) {
    return "log";
  } else {
    assert(false & "unsupported type for log op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::Log1pOp>(Type type) {
  if (type.isF32() || type.isF64()) {
    return "log1p";
  } else {
    assert(false & "unsupported type for abs op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::NegOp>(Type type) {
  return "(-)";
}

template <>
std::string CUDAMathFuncName<lmhlo::NotOp>(Type type) {
  return "(!)";
}

template <>
std::string CUDAMathFuncName<lmhlo::RsqrtOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hrsqrt";
  } else if (type.isF32() || type.isF64()) {
    return "rsqrt";
  } else {
    assert(false & "unsupported type for rsqrt op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::SqrtOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hsqrt";
  } else if (type.isF32() || type.isF64()) {
    return "sqrt";
  } else {
    assert(false & "unsupported type for sqrt op.");
  }
}

template <>
std::string CUDAMathFuncName<lmhlo::TanhOp>(Type type) {
  if (type.isF32() || type.isF64()) {
    return "tanh";
  } else {
    assert(false & "unsupported type for tanh op.");
  }
}

// TODO:
// template<lmhlo::SignOp>
// template<lmhlo::LogisticOp>

std::string MLIRType2CUDATypeStr(Type type) {
  if (type.isBF16()) {
    return "__nv_bfloat162";
  } else if (type.isF16()) {
    return "half";
  } else if (type.isF32()) {
    return "float";
  } else if (type.isF64()) {
    return "double";
  } else if (type.isSignedInteger(1)) {
    return "bool";
  } else if (type.isSignedInteger(32)) {
    return "int32_t";
  } else if (type.isUnsignedInteger(32)) {
    return "uint32_t";
  } else if (type.isIndex()) {
    return "size_t";
  } else if (type.isSignedInteger(64)) {
    return "int64_t";
  } else if (type.isUnsignedInteger(64)) {
    return "uint64_t";
  } else {
    assert(false & "unsupported type for convert op.");
  }
}

// Rely on nvcc to use fast-math.
llvm::Optional<std::string> SourceEmitterCUDA::EmitElemWiseUnaryOp(
    Operation* op, ValueNameBinding binding) {
  auto input = op->getOperand(0);
  if (binding.count(input) == 0) {
    llvm::None;
  }
  std::string input_str = binding[input];

  Type result_type =
      op->getOperand(1).getType().cast<MemRefType>().getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name;
  std::string expression;
  if (isa<lmhlo::AbsOp>(op)) {
    result_name = EmitUniqueName("abs");
    expression =
        CUDAMathFuncName<lmhlo::AbsOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::CeilOp>(op)) {
    result_name = EmitUniqueName("ceil");
    expression =
        CUDAMathFuncName<lmhlo::CeilOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::ConvertOp>(op)) {
    result_name = EmitUniqueName("convert");
    expression =
        CUDAMathFuncName<lmhlo::ConvertOp>(result_type) + "(" + input_str + ")";
    // } else if (isa<lmhlo::CopyOp>) {
  } else if (isa<lmhlo::CosineOp>(op)) {
    result_name = EmitUniqueName("consine");
    expression =
        CUDAMathFuncName<lmhlo::CosineOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::ExpOp>(op)) {
    result_name = EmitUniqueName("exp");
    expression =
        CUDAMathFuncName<lmhlo::ExpOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::FloorOp>(op)) {
    result_name = EmitUniqueName("floor");
    expression =
        CUDAMathFuncName<lmhlo::FloorOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::IsFiniteOp>(op)) {
    result_name = EmitUniqueName("isfinite");
    expression = CUDAMathFuncName<lmhlo::IsFiniteOp>(result_type) + "(" +
                 input_str + ")";
  } else if (isa<lmhlo::LogOp>(op)) {
    result_name = EmitUniqueName("log");
    expression =
        CUDAMathFuncName<lmhlo::LogOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::Log1pOp>(op)) {
    result_name = EmitUniqueName("log1p");
    expression =
        CUDAMathFuncName<lmhlo::Log1pOp>(result_type) + "(" + input_str + ")";
    // } else if (isa<lmhlo::LogisticOp) {
  } else if (isa<lmhlo::NegOp>(op)) {
    result_name = EmitUniqueName("neg");
    expression =
        CUDAMathFuncName<lmhlo::NegOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::NotOp>(op)) {
    result_name = EmitUniqueName("not");
    expression =
        CUDAMathFuncName<lmhlo::NotOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::RsqrtOp>(op)) {
    result_name = EmitUniqueName("rsqrt");
    expression =
        CUDAMathFuncName<lmhlo::RsqrtOp>(result_type) + "(" + input_str + ")";
    // } else if (isa<lmhlo::SignOp) {
  } else if (isa<lmhlo::SqrtOp>(op)) {
    result_name = EmitUniqueName("sqrt");
    expression =
        CUDAMathFuncName<lmhlo::SqrtOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::TanhOp>(op)) {
    result_name = EmitUniqueName("tanh");
    expression =
        CUDAMathFuncName<lmhlo::TanhOp>(result_type) + "(" + input_str + ")";
  }

  assert(binding.count(op->getOperand(1)) == 0);
  binding[op->getOperand(1)] = result_name;

  return type_str + " " + result_name + " = " + expression;
}

llvm::Optional<std::string> SourceEmitterCUDA::EmitElemWiseBinaryOp(
    Operation* op, ValueNameBinding binding) {
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  if (binding.count(lhs) == 0 || binding.count(rhs) == 0) {
    llvm::None;
  }
  std::string lhs_str = binding[lhs];
  std::string rhs_str = binding[rhs];

  Type result_type =
      op->getOperand(2).getType().cast<MemRefType>().getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name;
  std::string expression;
  if (isa<lmhlo::AddOp>(op)) {
    result_name = EmitUniqueName("add");
    expression = lhs_str + " + " + rhs_str;
  } else if (isa<lmhlo::SubtractOp>(op)) {
    result_name = EmitUniqueName("subtract");
    expression = lhs_str + " - " + rhs_str;
  } else if (isa<lmhlo::MulOp>(op)) {
    result_name = EmitUniqueName("mul");
    expression = lhs_str + " * " + rhs_str;
  } else if (isa<lmhlo::DivOp>(op)) {
    result_name = EmitUniqueName("div");
    expression = lhs_str + " / " + rhs_str;
  } else if (isa<lmhlo::MaxOp>(op)) {
    result_name = EmitUniqueName("max");
    expression = lhs_str + " > " + rhs_str + " ? " + lhs_str + " : " + rhs_str;
  } else if (isa<lmhlo::MinOp>(op)) {
    result_name = EmitUniqueName("min");
    expression = lhs_str + " < " + rhs_str + " ? " + lhs_str + " : " + rhs_str;
  } else if (auto compare = dyn_cast_or_null<lmhlo::CompareOp>(op)) {
    result_name = EmitUniqueName("compare");
    std::string cmp_str;
    switch (compare.getComparisonDirection()) {
      case mhlo::ComparisonDirection::EQ:
        cmp_str = "==";
        break;
      case mhlo::ComparisonDirection::NE:
        cmp_str = "!=";
        break;
      case mhlo::ComparisonDirection::LT:
        cmp_str = "<";
        break;
      case mhlo::ComparisonDirection::LE:
        cmp_str = "<=";
        break;
      case mhlo::ComparisonDirection::GT:
        cmp_str = ">";
        break;
      case mhlo::ComparisonDirection::GE:
        cmp_str = ">=";
        break;
    }
    expression = lhs_str + " " + cmp_str + " " + rhs_str;
  } else if (isa<lmhlo::AndOp>(op)) {
    result_name = EmitUniqueName("and");
    expression = lhs_str + " & " + rhs_str;
  } else if (isa<lmhlo::OrOp>(op)) {
    result_name = EmitUniqueName("or");
    expression = lhs_str + " | " + rhs_str;
  } else if (isa<lmhlo::RemOp>(op)) {
    result_name = EmitUniqueName("rem");
    expression = lhs_str + " % " + rhs_str;
  } else if (isa<lmhlo::PowOp>(op)) {
    result_name = EmitUniqueName("pow");
    expression = "pow(" + lhs_str + ", " + rhs_str + ")";
  }

  assert(binding.count(op->getOperand(2)) == 0);
  binding[op->getOperand(2)] = result_name;

  return type_str + " " + result_name + " = " + expression;
}

std::string SourceEmitterCUDA::EmitUniqueName(llvm::StringRef op_str) {
  std::string name = "bladedisc_" + op_str.str();
  if (existing_names_.count(op_str.str()) == 0) {
    existing_names_.try_emplace(op_str.str(), 0);
  }
  int32_t count = existing_names_[op_str.str()]++;
  name += std::to_string(count);

  return name;
}

llvm::Optional<std::string> SourceEmitterCUDA::EmitElemWiseTernaryOp(
    Operation* op, ValueNameBinding& binding) {
  auto input0 = op->getOperand(0);
  auto input1 = op->getOperand(1);
  auto input2 = op->getOperand(2);
  if (binding.count(input0) == 0 || binding.count(input1) == 0 ||
      binding.count(input2) == 0) {
    llvm::None;
  }
  std::string input0_str = binding[input0];
  std::string input1_str = binding[input1];
  std::string input2_str = binding[input2];

  Type result_type =
      op->getOperand(3).getType().cast<MemRefType>().getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name;
  std::string expression;
  if (isa<lmhlo::SelectOp>(op)) {
    result_name = EmitUniqueName("select");
    expression = input0_str + " ? " + input1_str + " : " + input2_str;
  } else if (isa<lmhlo::ClampOp>(op)) {
    result_name = EmitUniqueName("clamp");
    expression = input0_str + " < " + input1_str + " ? " + input1_str + " : (" +
                 input0_str + " > " + input2_str + " ? " + input2_str + " : " +
                 input0_str + ")";
  }

  assert(binding.count(op->getOperand(3)) == 0);
  binding[op->getOperand(3)] = result_name;

  return type_str + " " + result_name + " = " + expression;
}

void SourceEmitterCUDA::initValueNameBinding(SmallVectorImpl<Value> inputs,
                                             SmallVectorImpl<std::string> names,
                                             ValueNameBinding& binding) {
  assert(inputs.size() == names.size());
  for (int64_t i = 0; i < inputs.size(); i++) {
    binding[inputs[i]] = names[i];
  }
}

}  // namespace disc_ral
}  // namespace mlir
