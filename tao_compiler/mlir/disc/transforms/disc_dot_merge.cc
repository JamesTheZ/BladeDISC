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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/utils/cycle_detector.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/shape_utils.h"

#define DEBUG_TYPE "disc-dot-merge"

// TODO: this pass shares some common functions with lhlo-fusion-pass. The dot
// merge can actually be regarded as a kind of horizontal fusion. We will
// reassess the necessity of merging this pass into the fusion-pass after fixing
// the bugs in horizontal fusion functions in lhlo-fusion-pass.

namespace mlir {
namespace disc_ral {
namespace {

llvm::Optional<int32_t> TryMergeNode(GraphCycles* graph_cycles, int32_t a,
                                     int32_t b) {
  if (graph_cycles == nullptr) {
    return llvm::None;
  }
  bool has_edge_inserted_a2b = false;
  if (!graph_cycles->HasEdge(a, b) && !graph_cycles->HasEdge(b, a)) {
    has_edge_inserted_a2b = graph_cycles->InsertEdge(a, b);
    if (!has_edge_inserted_a2b) {
      // Cannot merge a and b as we cannot even insert an edge between a and b.
      return llvm::None;
    }
  }
  int32_t from = graph_cycles->HasEdge(a, b) ? a : b;
  int32_t to = (from == a) ? b : a;
  auto result = graph_cycles->ContractEdge(from, to);
  if (!result.hasValue() && has_edge_inserted_a2b) {
    // Restore the graph.
    graph_cycles->RemoveEdge(a, b);
  }
  return result;
}

// NOTE: this function is copied from `lhlo_fusion.cc`.
// Returns all the values touched by this op or its nested ops.
SmallVector<Value, 4> GetAllPossibleUsedValues(Operation* op) {
  SmallVector<Value, 4> values;
  op->walk([&](Operation* nest_op) {
    for (Value v : nest_op->getOperands()) {
      values.push_back(v);
    }
  });
  return values;
}

// Arrange the insert-point of `op`'s operands in the same block. Do not deal
// with cross-block operands.
void ArrangeOperandsInsertPointInBlock(Operation* op) {
  for (auto operand : GetAllPossibleUsedValues(op)) {
    auto operandOp = operand.getDefiningOp();
    // Note that `isBeforeInBlock` also check whether `op` and `operandOp` are
    // in the same block.
    if ((operandOp != nullptr) && !operandOp->isBeforeInBlock(op)) {
      operandOp->moveBefore(op);
      ArrangeOperandsInsertPointInBlock(operandOp);
    }
  }
}

void BuildBlockGraphCycles(Block* block,
                           std::unique_ptr<GraphCycles>& cycle_detector,
                           DenseMap<Operation*, int64_t>& op_to_node_id) {
  std::vector<Operation*> op_list;
  op_to_node_id.clear();
  for (Operation& op : *block) {
    op_to_node_id.try_emplace(&op, op_list.size());
    op_list.push_back(&op);
  }
  cycle_detector.reset(new GraphCycles(op_list.size()));
  for (int64_t node_id = 0; node_id < op_list.size(); node_id++) {
    Operation* op = op_list[node_id];
    for (Value operand : GetAllPossibleUsedValues(op)) {
      Operation* operand_op = operand.getDefiningOp();
      // Only consider the operand_op inside the target block.
      auto iter = op_to_node_id.find(operand_op);
      if (iter == op_to_node_id.end()) {
        continue;
      }
      cycle_detector->InsertEdge(iter->second, node_id);
    }
  }
}

struct DotCluster {
  DotCluster(Operation* op, int op_id) : leader_op_id(op_id) {
    ops.push_back(op);
  }

  // Merges `other` into this cluster, and clears `other`.
  void merge(DotCluster& other) {
    ops.insert(ops.end(), other.ops.begin(), other.ops.end());
    other.ops.clear();
  }

  // ID of the representative node of this cluster.
  int leader_op_id;

  // Dot ops to be merged.
  SmallVector<Operation*> ops;
};

bool TryMergeDotClusters(DotCluster& dst, DotCluster& src,
                         std::unique_ptr<GraphCycles>& cycle_detector) {
  // Check cycle.
  int64_t dst_id = dst.leader_op_id;
  int64_t src_id = src.leader_op_id;
  auto optional_merged_id = TryMergeNode(cycle_detector.get(), dst_id, src_id);
  if (!optional_merged_id.hasValue()) {
    // It forms a cycle.
    return false;
  }
  dst.merge(src);
  dst.leader_op_id = *optional_merged_id;
  return true;
}

// The `MergingShapeEqualityMap` is a map whose value is a list of dots with
// same `MergingShape`.
template <typename MergingShapeEqualityMap>
bool BuildDotClusters(Block* block,
                      const MergingShapeEqualityMap& equal_merge_shape_map,
                      SmallVector<DotCluster>& merging_clusters) {
  merging_clusters.clear();
  // Dot ops in `equal_merge_shape_map` can be merged together if the merging
  // does not introduce cycle.

  // Form cycle detector.
  std::unique_ptr<GraphCycles> cycle_detector(new GraphCycles(0));
  DenseMap<Operation*, int64_t> op_to_node_id;
  BuildBlockGraphCycles(block, cycle_detector, op_to_node_id);

  // Find merge clusters.
  for (auto& dots_can_merge : equal_merge_shape_map) {
    auto ops = dots_can_merge.second;
    if (ops.size() < 2) {
      continue;
    }
    SmallVector<DotCluster> clusters;
    for (auto op : ops) {
      clusters.emplace_back(op.getOperation(),
                            op_to_node_id[op.getOperation()]);
    }
    for (int64_t i = 0; i < clusters.size(); i++) {
      auto& merged = clusters[i];
      if (merged.ops.empty()) {
        continue;
      }
      for (int64_t j = i + 1; j < clusters.size(); j++) {
        auto& to_merge = clusters[j];
        if (to_merge.ops.empty()) {
          continue;
        }
        // Try merge.
        TryMergeDotClusters(merged, to_merge, cycle_detector);
      }
    }
    for (auto& cluster : clusters) {
      if (cluster.ops.size() > 1) {
        merging_clusters.push_back(cluster);
      }
    }
  }

  return true;
}

class DotMergingConverter {
 public:
  DotMergingConverter(FuncOp func) : func_(func){};
  bool run();

 public:
  struct MergingShape {
    Value dominant;
    mhlo::DotDimensionNumbersAttr dimension_numbers;
    bool operator==(const MergingShape& other) const {
      return (dominant == other.dominant) &&
             (dimension_numbers == other.dimension_numbers);
    }
  };

  struct MergingShapeHash {
    std::size_t operator()(const MergingShape& dotShape) const {
      std::size_t hash = mlir::hash_value(dotShape.dominant);

      const auto& dimension_numbers = dotShape.dimension_numbers;
      const auto& lhs_batch_dims = dimension_numbers.getLhsBatchingDimensions();
      const auto& rhs_batch_dims = dimension_numbers.getRhsBatchingDimensions();
      const auto& lhs_contracting_dims =
          dimension_numbers.getLhsContractingDimensions();
      const auto& rhs_contracting_dims =
          dimension_numbers.getRhsContractingDimensions();
      hash = llvm::hash_combine(hash,
                                llvm::hash_combine_range(lhs_batch_dims.begin(),
                                                         lhs_batch_dims.end()));
      hash = llvm::hash_combine(hash,
                                llvm::hash_combine_range(rhs_batch_dims.begin(),
                                                         rhs_batch_dims.end()));
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(lhs_contracting_dims.begin(),
                                         lhs_contracting_dims.end()));
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(rhs_contracting_dims.begin(),
                                         rhs_contracting_dims.end()));
      return hash;
    }
  };

  using MergingShapeEqualMap =
      std::unordered_map<MergingShape, SmallVector<mhlo::DotGeneralOp>,
                         MergingShapeHash>;

 private:
  bool buildMergingShapeMap(Block* block, ShapeAnalysis& analysis,
                            MergingShapeEqualMap& equal_merge_shape_map,
                            bool lhs_dominant);
  bool applyMerging(DotCluster& cluster, bool lhs_dominant);

 private:
  FuncOp func_;
};

bool DotMergingConverter::run() {
  ShapeAnalysis analysis(func_);
  if (failed(analysis.run())) {
    LLVM_DEBUG(llvm::dbgs() << "ShapeAnalysis failes for dot merge.\n");
    return false;
  }

  SmallVector<Block*> blocks;
  func_.walk([&](Block* block) { blocks.push_back(block); });

  for (Block* block : blocks) {
    for (auto lhs_dominant : SmallVector<bool, 2>({true, false})) {
      // A map to help to cluster dots with same shape and dim-numbers together.
      MergingShapeEqualMap equal_merge_shape_map;
      if (!buildMergingShapeMap(block, analysis, equal_merge_shape_map,
                                lhs_dominant)) {
        continue;
      }
      // Find merging clusters.
      SmallVector<DotCluster> merging_clusters;
      BuildDotClusters<MergingShapeEqualMap>(block, equal_merge_shape_map,
                                             merging_clusters);
      // Apply merging.
      for (auto& cluster : merging_clusters) {
        applyMerging(cluster, lhs_dominant);
      }
    }
  }

  return true;
}

bool DotMergingConverter::buildMergingShapeMap(
    Block* block, ShapeAnalysis& analysis,
    MergingShapeEqualMap& equal_merge_shape_map, bool lhs_dominant) {
  block->walk([&](mhlo::DotGeneralOp op) {
    MergingShape dot_shape;
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    dot_shape.dominant = lhs_dominant ? lhs : rhs;
    dot_shape.dimension_numbers = op.dot_dimension_numbers();

    auto& op_list = equal_merge_shape_map[std::move(dot_shape)];
    op_list.push_back(op);
  });
  return true;
}

bool DotMergingConverter::applyMerging(DotCluster& cluster, bool lhs_dominant) {
  auto& ops = cluster.ops;
  auto loc = ops.front()->getLoc();
  auto foremost = ops.front();
  for (int64_t i = 1; i < ops.size(); i++) {
    auto& op = ops[i];
    if (op->isBeforeInBlock(foremost)) {
      foremost = op;
    }
  }
  // Move all dot ops, and their consumers if necessary, before the original
  // foremost dot. This makes sure that the newly created ops in this function
  // dominates their uses.
  for (auto op : ops) {
    if (foremost == op) {
      continue;
    }
    op->moveBefore(foremost);
    ArrangeOperandsInsertPointInBlock(op);
  }

  auto last_dot = dyn_cast<mhlo::DotGeneralOp>(foremost);

  // Find concat dim.
  int64_t concat_dim = -1;
  SmallVector<Value> to_concat;
  for (auto op : ops) {
    to_concat.push_back(lhs_dominant ? op->getOperand(0) : op->getOperand(1));
  }
  DenseSet<int64_t> non_concat_dims;
  auto dim_numbers = last_dot.dot_dimension_numbers();
  // Batching and contracting dims of the to-concat operands.
  const auto& batch_dims = lhs_dominant
                               ? dim_numbers.getRhsBatchingDimensions()
                               : dim_numbers.getLhsBatchingDimensions();
  const auto& contract_dims = lhs_dominant
                                  ? dim_numbers.getRhsContractingDimensions()
                                  : dim_numbers.getLhsContractingDimensions();
  non_concat_dims.insert(batch_dims.begin(), batch_dims.end());
  non_concat_dims.insert(contract_dims.begin(), contract_dims.end());
  auto dominant_ty =
      lhs_dominant ? last_dot.lhs().getType().dyn_cast<RankedTensorType>()
                   : last_dot.rhs().getType().dyn_cast<RankedTensorType>();
  assert(dominant_ty);
  // Lhs and rhs have the same rank.
  int64_t rank = dominant_ty.getRank();
  assert(rank == non_concat_dims.size() + 1);
  for (int64_t i = 0; i < rank; i++) {
    if (!non_concat_dims.contains(i)) {
      concat_dim = i;
      break;
    }
  }
  assert(concat_dim > 0);

  // Build concat ranked-tensor-type and operands.
  SmallVector<int64_t, 4> concat_shape(rank, ShapedType::kDynamicSize);
  auto concat_operand0_ty =
      lhs_dominant
          ? ops[0]->getOperand(1).getType().dyn_cast<RankedTensorType>()
          : ops[0]->getOperand(0).getType().dyn_cast<RankedTensorType>();
  assert(concat_operand0_ty);
  for (int64_t i = 0; i < rank; i++) {
    // The non-concat dims are the same with any of the to-be-concatenated one.
    if (i != concat_dim) {
      concat_shape[i] = concat_operand0_ty.getDimSize(i);
    }
  }
  int64_t concat_dim_accu = 0;
  SmallVector<Value, 4> concat_operands;
  for (auto op : ops) {
    auto concat_operand = lhs_dominant ? op->getOperand(1) : op->getOperand(0);
    concat_operands.push_back(concat_operand);
    auto concat_operand_ty =
        lhs_dominant ? concat_operand.getType().dyn_cast<RankedTensorType>()
                     : concat_operand.getType().dyn_cast<RankedTensorType>();
    assert(concat_operand_ty);
    int64_t dim_size = concat_operand_ty.getDimSize(concat_dim);
    if (dim_size == ShapedType::kDynamicSize) {
      concat_dim_accu = ShapedType::kDynamicSize;
      break;
    } else {
      concat_dim_accu += dim_size;
    }
  }
  concat_shape[concat_dim] = concat_dim_accu;
  auto element_ty = dominant_ty.getElementType();
  auto concat_result_tp = RankedTensorType::get(concat_shape, element_ty);

  // Build concat op.
  OpBuilder builder(last_dot);
  Value concat = builder.create<mhlo::ConcatenateOp>(
      loc, concat_result_tp, concat_operands,
      builder.getI64IntegerAttr(concat_dim));

  // Build result type.
  auto orig0_result_ty =
      ops[0]->getResult(0).getType().dyn_cast<RankedTensorType>();
  SmallVector<int64_t, 4> result_shape(rank, ShapedType::kDynamicSize);
  for (int64_t i = 0; i < rank; i++) {
    result_shape[i] = orig0_result_ty.getDimSize(i);
  }
  // According to `DotGeneralOp::reifyReturnTypeShapes`, m and n dimes are the
  // last two dimensions.
  int64_t concat_dim_in_result = lhs_dominant ? (rank - 1) : (rank - 2);
  result_shape[concat_dim_in_result] = concat_dim_accu;
  auto dot_result_type = RankedTensorType::get(result_shape, element_ty);

  // Create Dot op.
  auto dot_dimension_attr = mhlo::DotDimensionNumbersAttr::get(
      builder.getContext(), dim_numbers.getLhsBatchingDimensions(),
      dim_numbers.getRhsBatchingDimensions(),
      dim_numbers.getLhsContractingDimensions(),
      dim_numbers.getRhsContractingDimensions());
  auto lhs = lhs_dominant ? ops[0]->getOperand(0) : concat;
  auto rhs = lhs_dominant ? concat : ops[0]->getOperand(1);
  Value merged_dot = builder.create<mhlo::DotGeneralOp>(
      loc, dot_result_type, lhs, rhs, dot_dimension_attr, nullptr);

  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  // Build slice and reshape op for each of the original dot op, and replace the
  // dot.
  if (dot_result_type.getNumDynamicDims() == 0) {
    int64_t start = 0;
    int64_t limit;
    for (int64_t i = 0; i < ops.size(); i++) {
      mhlo::DotGeneralOp op = dyn_cast<mhlo::DotGeneralOp>(ops[i]);
      // Starts.
      SmallVector<int64_t> starts(rank, 0);
      starts[concat_dim_in_result] = start;
      // Limits.
      auto concat_operand = lhs_dominant ? op.rhs() : op.lhs();
      auto operand_ty = concat_operand.getType().dyn_cast<RankedTensorType>();
      // Note that concat_dim is for operand, while concat_dim_in_result is for
      // the merged dot.
      limit = start + operand_ty.getDimSize(concat_dim);
      SmallVector<int64_t> limits(rank);
      for (int64_t i = 0; i < rank; i++) {
        if (i != concat_dim_in_result) {
          limits[i] = dot_result_type.getDimSize(i);
        } else {
          limits[i] = limit;
        }
      }
      // Strides.
      SmallVector<int64_t> strides(rank, 1);
      // Create slice op and replace uses.
      auto slice = builder.create<mhlo::SliceOp>(
          loc, merged_dot, GetI64ElementsAttr(starts, &builder),
          GetI64ElementsAttr(limits, &builder),
          GetI64ElementsAttr(strides, &builder));
      op->replaceAllUsesWith(slice);

      // Update `start` for the next concatenated operand.
      start = limit;
    }
  } else {
    Value start = zero;
    Value limit;
    for (int64_t i = 0; i < ops.size(); i++) {
      mhlo::DotGeneralOp op = dyn_cast<mhlo::DotGeneralOp>(ops[i]);
      // Starts.
      SmallVector<Value, 4> starts(rank, zero);
      starts[concat_dim_in_result] = start;
      // Limits.
      auto concat_operand = lhs_dominant ? op.rhs() : op.lhs();
      limit = builder.create<arith::AddIOp>(
          loc, start,
          builder.create<tensor::DimOp>(loc, concat_operand, concat_dim));
      SmallVector<Value> limits(rank);
      for (int64_t i = 0; i < rank; i++) {
        if (i != concat_dim_in_result) {
          limits[i] = builder.create<tensor::DimOp>(loc, merged_dot, i);
        } else {
          limits[i] = limit;
        }
      }
      // Strides
      SmallVector<Value, 4> strides(rank, one);

      // Build start/limit/stride indices and slice type.
      auto index_ty = builder.getIndexType();
      auto start_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(starts.size())},
                                index_ty),
          starts);
      auto limit_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(limits.size())},
                                index_ty),
          limits);
      auto stride_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(strides.size())},
                                index_ty),
          strides);
      SmallVector<int64_t, 4> slice_shape(rank, ShapedType::kDynamicSize);
      for (int64_t i = 0; i < rank; i++) {
        if (i != concat_dim_in_result) {
          slice_shape[i] = dot_result_type.getDimSize(i);
        } else {
          auto operand_ty =
              concat_operand.getType().dyn_cast<RankedTensorType>();
          slice_shape[i] = operand_ty.getDimSize(concat_dim);
        }
      }
      auto slice_type = RankedTensorType::get(slice_shape, element_ty);

      // Create dyn-slice op and replace uses.
      auto dyn_slice = builder.create<mhlo::RealDynamicSliceOp>(
          loc, slice_type, merged_dot, start_indices, limit_indices,
          stride_indices);
      op->replaceAllUsesWith(dyn_slice);

      // Update `start` for the next concatenated operand.
      start = limit;
    }
  }

  // No longer need the original dot ops.
  for (int64_t i = 0; i < ops.size(); i++) {
    ops[i]->erase();
  }

  return true;
}

struct DiscDotMergePass : public DiscDotMergePassBase<DiscDotMergePass> {
  DiscDotMergePass()
      : DiscDotMergePassBase<DiscDotMergePass>::DiscDotMergePassBase() {}
  void runOnOperation() override;

 private:
  bool doSameOperandDotMerging(FuncOp& func);
};

void DiscDotMergePass::runOnOperation() {
  FuncOp func = getOperation();

  if (!doSameOperandDotMerging(func)) {
    signalPassFailure();
  }
}

bool DiscDotMergePass::doSameOperandDotMerging(FuncOp& func) {
  return DotMergingConverter(func).run();
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscDotMergePass() {
  return std::make_unique<DiscDotMergePass>();
}

}  // namespace disc_ral
}  // namespace mlir