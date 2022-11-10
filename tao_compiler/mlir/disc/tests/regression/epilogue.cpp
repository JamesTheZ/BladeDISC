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

#include "tensorflow/compiler/mlir/disc/tests/mlir_feature_test.h"
#include "tensorflow/compiler/mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"

namespace mlir_test {

const std::string c_ft_path =
    "tensorflow/compiler/mlir/disc/tests/regression/data/";

// GELU f16.
TEST(EpilogueTest, EpilogueGELUF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "epilogue_gelu_f16.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x16x128x768xf16_X", "1x16x768x768xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// GELU f32.
TEST(EpilogueTest, EpilogueGELUF32) {
  // #if 0
  //   std::vector<float> A;
  //   for (int64_t i = 0; i < 1 * 16 * 128 * 768; i++) {
  //     // float val = i / (128 * 768) * 1000 * 1000 +
  //     // A.push_back(i);
  //     A.push_back(1.0);
  //   }
  //   std::vector<float> B;
  //   for (int64_t i = 0; i < 1 * 16 * 768 * 768; i++) {
  //     // float val = i / (128 * 768) * 1000 * 1000 +
  //     // B.push_back(i);
  //     B.push_back(1.0);
  //   }
  // #endif
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "epilogue_gelu_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x16x128x768xf32_X", "1x16x768x768xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  // {"f32_X"}, {A, B}));
}

}  // namespace mlir_test
