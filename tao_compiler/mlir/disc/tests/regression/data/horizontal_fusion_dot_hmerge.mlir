module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>)
      -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>)
      attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:3 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Abs"(%arg1) : (tensor<?x?xf16>) -> (tensor<?x?xf16>)
      %1:2 = tf_executor.island wraps "tf.Neg"(%0) : (tensor<?x?xf16>) -> (tensor<?x?xf16>)
      %2:2 = tf_executor.island wraps "tf.Rsqrt"(%1) : (tensor<?x?xf16>) -> (tensor<?x?xf16>)
      %3:2 = tf_executor.island wraps "tf.BatchMatMul"(%arg0, %0) {adj_x = false, adj_y = false} : (tensor<?x?xf16>, tensor<?x?xf16>) -> (tensor<?x?xf16>)
      %4:2 = tf_executor.island wraps "tf.BatchMatMul"(%arg0, %1) {adj_x = false, adj_y = false} : (tensor<?x?xf16>, tensor<?x?xf16>) -> (tensor<?x?xf16>)
      %5:2 = tf_executor.island wraps "tf.BatchMatMul"(%arg0, %2) {adj_x = false, adj_y = false} : (tensor<?x?xf16>, tensor<?x?xf16>) -> (tensor<?x?xf16>)
      tf_executor.fetch %3, %4, %5 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>
    }
    return %graph#0, %graph#1, %graph#2 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>
  }
}