// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
// #include "mlir-hlo/utils/codegen_utils.h"
// #include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
// #include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
// #include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
// #include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"
// #include "tensorflow/core/util/env_var.h"

namespace mlir {
namespace disc_ral {
namespace {

// using lmhlo::FusionOp;

// struct DiscStripShapeConstraintOpsPass
//     : public DiscStripShapeConstraintOpsPassBase<
//           DiscStripShapeConstraintOpsPass> {
//   void runOnOperation() override {
//     SmallVector<disc_shape::SymbolicDimOp> ops;
//     getOperation().walk(
//         [&](disc_shape::SymbolicDimOp op) { ops.push_back(op); });
//     for (disc_shape::SymbolicDimOp op : ops) {
//       op->erase();
//     }
//     StringRef funcName =
//     SymbolicDimMgr::getShapeConstraintGraphFunctionName();
//     // first try to remove the old shape constraint graph
//     if (auto func = getOperation().lookupSymbol<func::FuncOp>(funcName))
//       func->erase();
//   }
// };

struct DiscDotFusionToFuncPass
    : public DiscDotFusionToFuncPassBase<DiscDotFusionToFuncPass> {
 public:
  // explicit DiscDotFusionToFuncPass()
  //     : DiscDotFusionToFuncPassBase<DiscDotFusionToFuncPass>::
  //           DiscDotFusionToFuncPassBase() {}
  void runOnOperation() override;

 private:
  void convertKDotFusionToFunc(lmhlo::FusionOp op);
};

void DiscDotFusionToFuncPass::runOnOperation() {
  ModuleOp module_op = getOperation();

  SmallVector<lmhlo::FusionOp> kdot_fusions;
  module_op.walk([&](lmhlo::FusionOp fusion) {
    if (isFusionType<FusionType::kDot>(fusion)) {
      kdot_fusions.push_back(fusion);
    }
  });

  for (auto op : kdot_fusions) {
    convertKDotFusionToFunc(op);
  }
}

void DiscDotFusionToFuncPass::convertKDotFusionToFunc(lmhlo::FusionOp op) {
  auto parent_func = op->getParentOfType<func::FuncOp>();
  OpBuilder builder(parent_func);
  Location loc = parent_func.getLoc();

  FusionPatternBase fusion_pattern(op);
  const auto& operands = fusion_pattern.getOperands();
  const auto& results = fusion_pattern.getResults();
  const auto& intermediate_buffers = fusion_pattern.getInternalResults();

  // Prepare func type.
  SmallVector<Value> func_operands;
  SmallVector<Type> func_operand_types;
  for (Value operand : operands) {
    func_operands.push_back(operand);
    func_operand_types.push_back(operand.getType());
  }
  for (Value result : results) {
    func_operands.push_back(result);
    func_operand_types.push_back(result.getType());
  }
  for (Value buffer : intermediate_buffers) {
    func_operands.push_back(buffer);
    func_operand_types.push_back(buffer.getType());
  }
  FunctionType type =
      FunctionType::get(builder.getContext(), func_operand_types, {});

  // Create func op and clone instructions.
  auto fusion_func = builder.create<func::FuncOp>(loc, getFusionName(op), type);
  BlockAndValueMapping mapper;
  builder.setInsertionPointToEnd(fusion_func.addEntryBlock());
  for (const auto& arg : enumerate(fusion_func.getArguments())) {
    if (arg.index() < operands.size()) {
      mapper.map(operands[arg.index()], arg.value());
    } else if (arg.index() < operands.size() + results.size()) {
      mapper.map(results[arg.index() - operands.size()], arg.value());
    } else {
      mapper.map(
          intermediate_buffers[arg.index() - operands.size() - results.size()],
          arg.value());
    }
  }
  for (auto& inst : op.getRegion().getOps()) {
    if (isa<lmhlo::TerminatorOp>(&inst)) {
      continue;
    }
    builder.clone(inst, mapper);
  }

  builder.create<func::ReturnOp>(loc);

  fusion_func->setAttr(kFuncDotFusionAttr,
                       builder.getIntegerAttr(builder.getIntegerType(1), 1));

  // Create call op.
  OpBuilder builder_call(op);
  auto call = builder_call.create<func::CallOp>(loc, fusion_func.getName(),
                                                TypeRange({}), func_operands);
  op.erase();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscDotFusionToFuncPass() {
  return std::make_unique<DiscDotFusionToFuncPass>();
}

}  // namespace disc_ral
}  // namespace mlir
