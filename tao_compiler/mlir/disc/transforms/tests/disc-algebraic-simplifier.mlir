// RUN: disc-opt -disc-algebraic-simplifier -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func @dot_batching
func @dot_batching(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "mhlo.abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.abs"(%arg1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.dot_general"(%2, %3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.add"(%1, %4) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: mhlo.dot_general
  // CHECK-DAG: lhs_batching_dimensions = [0]
  // CHECK-DAG: lhs_contracting_dimensions = [2]
  // CHECK-DAG: rhs_batching_dimensions = [0]
  // CHECK-DAG: rhs_contracting_dimensions = [1]
  // CHECK-NOT: mhlo.dot_general
  return %5: tensor<?x?xf32>
}

// CHECK-LABEL: func @dot_not_batching_diff_dtype
func @dot_not_batching_diff_dtype(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "mhlo.convert"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  %3 = "mhlo.convert"(%arg1) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  %4 = "mhlo.dot_general"(%2, %3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %5 = "mhlo.convert"(%4) : (tensor<?x?xf16>) -> tensor<?x?xf32>
  %6 = "mhlo.add"(%1, %5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [1]
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [1]
  return %5: tensor<?x?xf32>
}

// CHECK-LABEL: func @dot_not_batching_cycle
func @dot_not_batching_cycle(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "mhlo.abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.abs"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.dot_general"(%2, %3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.add"(%1, %4) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [1]
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [1]
  return %5: tensor<?x?xf32>
}
