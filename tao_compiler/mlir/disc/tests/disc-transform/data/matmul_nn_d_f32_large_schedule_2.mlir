transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation

  %0:2 = transform.structured.tile_to_forall_op %matmul num_threads [1, 1]
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  transform.structured.fuse_into_containing_op %fill into %0#0
    : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

  // further tile and fuse matmul and fill op.
  %1:3 = transform.structured.fuse %0#1 {tile_sizes = [6, 16, 0], tile_interchange = [0, 1, 2]}

  // gemm reduction axis tiling
  %2:2 = transform.structured.tile %1#0 [0, 0, 1] {interchange=[0, 1, 2]}  : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

  // do some basic cleanup.
  %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.disc.apply_patterns %func0 {canonicalization}

  // pad to match the requirement of hardware vector/tensor instruction.
  %3 = transform.structured.pad %2#0 {padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32], padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 0], hoist_paddings=[2, 3, 0], transpose_paddings=[[1, 0], [0, 1], [0, 1]]}
    : (!pdl.operation) -> !pdl.operation

  %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.vectorize %func1 {vectorize_padding}

  %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.disc.apply_patterns %func2 {canonicalization}

  %arg2 = transform.disc.bufferize %arg1
  transform.disc.vector.lower_vectors %arg2 contraction_lowering = outerproduct multireduction_lowering = innerparallel split_transfers = "linalg-copy" transpose_lowering = eltwise
}