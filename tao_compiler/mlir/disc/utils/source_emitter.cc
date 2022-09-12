
#include "tensorflow/compiler/mlir/disc/utils/source_emitter.h"

namespace mlir {
namespace disc_ral {

using ValueNameBinding = SourceEmitterCUDA::ValueNameBinding;

template <typename OP>
std::string CUDAMathFuncName(Type type) {
  assert(false & "unsupported op");
}

template <lmhlo::AbsOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "__habs";
  } else if (type.isF32() || type.isF64() || type.isInteger(32) ||
             type.isInteger(64)) {
    return "abs";
  } else {
    assert(false & "unsupported type for abs op.");
  }
}

template <lmhlo::CeilOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hceil";
  } else if (type.isF32() || type.isF64()) {
    return "ceil";
  } else {
    assert(false & "unsupported type for ceil op.");
  }
}

template <lmhlo::ConvertOp>
std::string CUDAMathFuncName(Type type) {
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

template <lmhlo::CosineOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hcos";
  } else if (type.isF32() || type.isF64()) {
    return "cos";
  } else {
    assert(false & "unsupported type for cosine op.");
  }
}

template <lmhlo::ExpOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hexp";
  } else if (type.isF32() || type.isF64()) {
    return "exp";
  } else {
    assert(false & "unsupported type for exp op.");
  }
}

template <lmhlo::FloorOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hfloor";
  } else if (type.isF32() || type.isF64()) {
    return "floor";
  } else {
    assert(false & "unsupported type for floor op.");
  }
}

template <lmhlo::IsFiniteOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isF32() || type.isF64()) {
    return "isfinite";
  } else {
    assert(false & "unsupported type for isfinite op.");
  }
}

template <lmhlo::LogOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hlog";
  } else if (type.isF32() || type.isF64()) {
    return "log";
  } else {
    assert(false & "unsupported type for log op.");
  }
}

template <lmhlo::Log1pOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isF32() || type.isF64()) {
    return "log1p";
  } else {
    assert(false & "unsupported type for abs op.");
  }
}

// template<lmhlo::LogisticOp>

template <lmhlo::NegOp>
std::string CUDAMathFuncName(Type type) {
  return "(-)";
}

template <lmhlo::NotOp>
std::string CUDAMathFuncName(Type type) {
  return "(!)";
}

template <lmhlo::RsqrtOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hrsqrt";
  } else if (type.isF32() || type.isF64()) {
    return "rsqrt";
  } else {
    assert(false & "unsupported type for rsqrt op.");
  }
}

// template<lmhlo::SignOp>

template <lmhlo::SqrtOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hsqrt";
  } else if (type.isF32() || type.isF64()) {
    return "sqrt";
  } else {
    assert(false & "unsupported type for sqrt op.");
  }
}

template <lmhlo::TanhOp>
std::string CUDAMathFuncName(Type type) {
  if (type.isF32() || type.isF64()) {
    return "tanh";
  } else {
    assert(false & "unsupported type for tanh op.");
  }
}

std::string SourceEmitterCUDA::EmitTypeStr(Type type) {
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
  auto lhs = op->getOperand(0);
  if (binding.count(lhs) == 0) {
    return nullptr;
  }
  std::string lhs_str = binding[lhs];

  Type type = op->getOperand(1).getType().cast<MemRefType>().getElementType();
  std::string type_str = EmitTypeStr(type);
  std::string result_name;
  std::string expression;
  if (isa < lmhlo::AbsOp) {
    result_name = EmitUniqueName("abs");
    expression =
        std::format(CUDAMathFuncName<lmhlo::AbsOp>(type_str) + "({})", lhs_str);
  } else if (isa < lmhlo::CeilOp) {
    result_name = EmitUniqueName("ceil");
    expression = std::format(CUDAMathFuncName<lmhlo::CeilOp>(type_str) + "({})",
                             lhs_str);
  } else if (isa < lmhlo::ConvertOp) {
    result_name = EmitUniqueName("convert");
    expression = std::format(
        CUDAMathFuncName<lmhlo::ConvertOp>(type_str) + "({})", lhs_str);
    // } else if (isa<lmhlo::CopyOp) {
  } else if (isa < lmhlo::CosineOp) {
    result_name = EmitUniqueName("consine");
    expression = std::format(
        CUDAMathFuncName<lmhlo::CosineOp>(type_str) + "({})", lhs_str);
  } else if (isa < lmhlo::ExpOp) {
    result_name = EmitUniqueName("exp");
    expression =
        std::format(CUDAMathFuncName<lmhlo::ExpOp>(type_str) + "({})", lhs_str);
  } else if (isa < lmhlo::FloorOp) {
    result_name = EmitUniqueName("floor");
    expression = std::format(
        CUDAMathFuncName<lmhlo::FloorOp>(type_str) + "({})", lhs_str);
  } else if (isa < lmhlo::IsFiniteOp) {
    result_name = EmitUniqueName("isfinite");
    expression = std::format(
        CUDAMathFuncName<lmhlo::IsFiniteOp>(type_str) + "({})", lhs_str);
  } else if (isa < lmhlo::LogOp) {
    result_name = EmitUniqueName("log");
    expression =
        std::format(CUDAMathFuncName<lmhlo::LogOp>(type_str) + "({})", lhs_str);
  } else if (isa < lmhlo::Log1pOp) {
    result_name = EmitUniqueName("log1p");
    expression = std::format(
        CUDAMathFuncName<lmhlo::Log1pOp>(type_str) + "({})", lhs_str);
    // } else if (isa<lmhlo::LogisticOp) {
  } else if (isa < lmhlo::NegOp) {
    result_name = EmitUniqueName("neg");
    expression =
        std::format(CUDAMathFuncName<lmhlo::NegOp>(type_str) + "({})", lhs_str);
  } else if (isa < lmhlo::NotOp) {
    result_name = EmitUniqueName("not");
    expression =
        std::format(CUDAMathFuncName<lmhlo::NotOp>(type_str) + "({})", lhs_str);
  } else if (isa < lmhlo::RsqrtOp) {
    result_name = EmitUniqueName("rsqrt");
    expression = std::format(
        CUDAMathFuncName<lmhlo::RsqrtOp>(type_str) + "({})", lhs_str);
    // } else if (isa<lmhlo::SignOp) {
  } else if (isa < lmhlo::SqrtOp) {
    result_name = EmitUniqueName("sqrt");
    expression = std::format(CUDAMathFuncName<lmhlo::SqrtOp>(type_str) + "({})",
                             lhs_str);
  } else if (isa < lmhlo::TanhOp) {
    result_name = EmitUniqueName("tanh");
    expression = std::format(CUDAMathFuncName<lmhlo::TanhOp>(type_str) + "({})",
                             lhs_str);
  }

  assert(binding.count(op->getOperand(1)) == 0);
  binding[op->getOperand(1)] = result_name;

  return std::format("{} {} = {};", type_str, result_name, expression);
}

llvm::Optional<std::string> SourceEmitterCUDA::EmitElemWiseBinaryOp(
    Operation* op, ValueNameBinding binding) {
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  if (binding.count(lhs) == 0 || binding.count(rhs) == 0) {
    return nullptr;
  }
  std::string lhs_str = binding[lhs];
  std::string rhs_str = binding[rhs];

  Type type = op->getOperand(1).getType().cast<MemRefType>().getElementType();
  std::string type_str = EmitTypeStr(op);
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
    switch (compare.comparison_direction()) {
      case ComparisonDirection::EQ:
        cmp_str = "==";
        break;
      case ComparisonDirection::NE:
        cmp_str = "!=";
        break;
      case ComparisonDirection::LT:
        cmp_str = "<";
        break;
      case ComparisonDirection::LE:
        cmp_str = "<=";
        break;
      case ComparisonDirection::GT:
        cmp_str = ">";
        break;
      case ComparisonDirection::GE:
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
  std::string name = "bladedisc_" + op_str;
  if (existing_names.count(op_str) == 0) {
    existing_names.emplace(op_str, 0);
  }
  int32_t count = existing_names[op_str]++;
  name += std::to_string(count);

  return name;
}

std::string SourceEmitterCUDA::EmitElemWiseTernaryOp(
    Operation* op, ValueNameBinding& binding) {
  /*
    if (isa<lmhlo::SelectOp, lmhlo::ClampOp>(op)) {
      auto ref = op->getOperand(0).getType().cast<MemRefType>();
      for (Value v : op->getOperands().drop_front())
        if (v.getType().cast<MemRefType>().getRank() != ref.getRank())
          return false;
      return true;
    }
  */
}

}  // namespace disc_ral
}  // namespace mlir

/*

// Returns true if the op is an elementwise unary lmhlo op.
// TODO(disc): use fusibility interface
// TODO(disc): Unify with disc_supported_list.h and Elementwise Trait
bool isElementWiseUnary(Operation* op) {
  // clang-format off
  return isa<
    lmhlo::AbsOp,
    lmhlo::CeilOp,
    lmhlo::ConvertOp,
    lmhlo::CopyOp,
    lmhlo::CosineOp,
    lmhlo::ExpOp,
    lmhlo::FloorOp,
    lmhlo::IsFiniteOp,
    lmhlo::LogOp,
    lmhlo::Log1pOp,
    lmhlo::LogisticOp,
    lmhlo::NegOp,
    lmhlo::NotOp,
    lmhlo::RsqrtOp,
    lmhlo::SignOp,
    lmhlo::SqrtOp,
    lmhlo::TanhOp
  >(op);
  // clang-format on
}

// Returns true if the op is an elementwise binary lmhlo op.
// TODO(disc): use fusibility interface
bool isElementWiseBinary(Operation* op) {
  // clang-format off
  return isa<
    lmhlo::AddOp,
    lmhlo::AndOp,
    lmhlo::CompareOp,
    lmhlo::DivOp,
    lmhlo::MaxOp,
    lmhlo::MinOp,
    lmhlo::MulOp,
    lmhlo::OrOp,
    lmhlo::PowOp,
    lmhlo::RemOp,
    lmhlo::SubtractOp
  >(op);
  // clang-format on
}

// Returns true if the op is an elementwise ternary lmhlo op.
bool isElementWiseTernary(Operation* op) {
  // Some ternary lmhlo ops (e.g. select op) suppport a restricted implicit
  // broadcast semantic, that is: if one input has rank zero, then it will be
  // automatically broadcasted if necessary. Thus we can know that there is no
  // broadcast if all operands have the same rank.
  if (isa<lmhlo::SelectOp, lmhlo::ClampOp>(op)) {
    auto ref = op->getOperand(0).getType().cast<MemRefType>();
    for (Value v : op->getOperands().drop_front())
      if (v.getType().cast<MemRefType>().getRank() != ref.getRank())
        return false;
    return true;
  }

  return false;
}
*/

#endif  // MLIR_DISC_UTILS_SOURCE_EMITTER