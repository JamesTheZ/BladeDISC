// RUN: disc-opt -disc-dot-merge -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func @dot_merging
// This UT builds the tensor shape implicitly because current implement of
// ShapeAnalysis relies on them to build DimValue. ShapeAnalysisV2, which is on
// going for development, will solve this problem.
func @dot_merging(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %m: tensor<index>, %n: tensor<index>, %k: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_m= tensor.extract %m[] : tensor<index>
  %dim_n = tensor.extract %n[] : tensor<index>
  %dim_k = tensor.extract %k[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim_m, %dim_k : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim_k, %dim_n : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.abs"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.dot_general"(%0, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK:     mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: mhlo.dot_general
  // CHECK:     mhlo.real_dynamic_slice
  // CHECK:     mhlo.real_dynamic_slice
  return %5: tensor<?x?xf32>
}

// CHECK-LABEL: func @dot_merging_static
func @dot_merging_static(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  %2 = "mhlo.abs"(%arg1) : (tensor<256x512xf32>) -> tensor<256x512xf32>
  %3 = "mhlo.dot_general"(%arg0, %2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  %4 = "mhlo.add"(%0, %3) : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
  // CHECK:     mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK:     -> tensor<128x1024xf32>
  // CHECK-NOT: mhlo.dot_general
  // CHECK:     mhlo.slice
  // CHECK-NOT: mhlo.real_dynamic_slice
  return %4: tensor<128x512xf32>
}

// CHECK-LABEL: func @dot_merging_partial_static
func @dot_merging_partial_static(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %m: tensor<index>, %n: tensor<index>, %k: tensor<index>) -> (tensor<?x?xf32>, tensor<?x512xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_m= tensor.extract %m[] : tensor<index>
  %dim_n = tensor.extract %n[] : tensor<index>
  %dim_k = tensor.extract %k[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim_m, %dim_k : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim_k, %dim_n : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %c512 = arith.constant 512 : index
  %partial_dyn_shape = tensor.from_elements %dim_k, %c512 : tensor<2xindex>
  %c1.0 = mhlo.constant dense<1.0> : tensor<f32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%c1.0, %partial_dyn_shape) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x512xf32>
  %5 = "mhlo.dot_general"(%0, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x512xf32>) -> tensor<?x512xf32>
  // CHECK:     mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: mhlo.dot_general
  // CHECK:     mhlo.real_dynamic_slice
  // CHECK:     -> tensor<?x?xf32>
  // CHECK:     mhlo.real_dynamic_slice
  // CHECK:     -> tensor<?x512xf32>
  return %2, %5: tensor<?x?xf32>, tensor<?x512xf32>
}

// CHECK-LABEL: func @dot_not_merging_cycle
func @dot_not_merging_cycle(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.extract %arg2[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim, %dim : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim, %dim : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.abs"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.dot_general"(%0, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NOT: mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: slice
  return %5: tensor<?x?xf32>
}
