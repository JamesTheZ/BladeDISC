// RUN: disc-opt %s -disc-memref-gvn | FileCheck %s

// CHECK-LABEL: @gvn_in_the_same_block
// CHECK-SAME: (%[[INPUT:.*]]: f32
func @gvn_in_the_same_block(%input: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 1 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK-NOT: memref.load
  %a = memref.load %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[INPUT]], %[[INPUT]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b : f32
}

// -----

// CHECK-LABEL: @gvn_in_the_dominant_block
// CHECK-SAME: (%[[INPUT:.*]]: f32
func @gvn_in_the_dominant_block(%input: f32, %pred: i1) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: scf.if
  %result = scf.if %pred -> (f32) {
    // CHECK-NOT: memref.load
    %b = memref.load %memref[%c0] : memref<?xf32, "cpu">
    // CHECK: scf.yield %[[INPUT]]
    scf.yield %b : f32 
  } else {
    %f0 = arith.constant 0.0 : f32
    scf.yield %f0 : f32
  }
  return %result : f32
}

// -----

// CHECK-LABEL: @should_not_gvn_diff_index
func @should_not_gvn_diff_index(%input: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 1 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL:.*]] = memref.load
  %a = memref.load %memref[%c1] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL]], %[[LOAD_VAL]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b : f32
}

// -----

// CHECK-LABEL: @should_not_gvn_diff_memref
func @should_not_gvn_diff_memref(%input: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 1 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  %memref_2 = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL:.*]] = memref.load
  %a = memref.load %memref_2[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL]], %[[LOAD_VAL]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b : f32
}

// -----


// CHECK-LABEL: @should_not_gvn_not_dominate
// CHECK-SAME: (%[[INPUT:.*]]: f32
func @should_not_gvn_not_dominate(%input: f32, %pred: i1) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  // CHECK: scf.if
  scf.if %pred -> () {
    // CHECK: memref.store
    memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
    scf.yield
  }
  // CHECK: memref.load
  %b = memref.load %memref[%c0] : memref<?xf32, "cpu">
  return %b: f32
}

// ----

// CHECK-LABEL: @should_not_gvn_on_gpu
func @should_not_gvn_on_gpu(%input: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 1 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "gpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "gpu">
  // CHECK: %[[LOAD_VAL:.*]] = memref.load
  %a = memref.load %memref[%c0] : memref<?xf32, "gpu">
  // CHECK: %[[LOAD_VAL]], %[[LOAD_VAL]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b : f32
}