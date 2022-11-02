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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/utils/source_emitter.h"

namespace mlir {
namespace disc_ral {
namespace {

constexpr const char* kSpecializedClassName = "SpecializedGemmFusion";
constexpr const char* kSpecializedEpilogue = "SpecializedEpilogue";
constexpr const char* kEpilogueIsHeavy = "EpilogueIsHeavy";

constexpr const char* kGRank = "GRank";
constexpr const char* kElementAType = "ElementAType";
constexpr const char* kElementALayout = "ElementALayout";
constexpr const char* kElementBType = "ElementBType";
constexpr const char* kElementBLayout = "ElementBLayout";
constexpr const char* kElementOutputType = "ElementOutputType";
constexpr const char* kElementOutputLayout = "ElementOutputLayout";
constexpr const char* kElementAccumulatorType = "ElementAccumulatorType";
constexpr const char* kOperatorClassType = "OperatorClassType";
constexpr const char* kSMArch = "SMArch";

constexpr const char* kScaleKind = "EpilogueScaleKind";
constexpr const char* kCountVectorized = "EpilogueCountVectorized";
constexpr const char* kEpilogueType = "EpilogueElementType";

constexpr const char* kGatherA = "IsGatherA";
constexpr const char* kGatherB = "IsGatherB";
constexpr const char* kScatterD = "IsScatterD";
constexpr const char* kPermuteDLayout = "EpiloguePermuteDLayout";

constexpr const char* kGemmFusionFuncName = "gemmFusionFunc";

// clang-format off
// The template is CUTLASS calling after preprocessing.
constexpr const char* gemm_fusion_template =
    "\n"
    "class SpecializedGemmFusion {\n"
    " public:\n"
    "  template <typename T>\n"
    "  struct ElementwiseUnaryOp {\n"
    "    __inline__ __attribute__((always_inline)) __attribute__((device)) __attribute__((host))\n"
    "    T operator()(T const& input) const {\n"
    "      return input;\n"
    "    }\n"
    "  };\n"
    "\n"
    "  template <typename T>\n"
    "  struct EpilogueFunctor {\n"
    "    static const bool kIsHeavy = EpilogueIsHeavy;\n"
    "\n"
    "    __inline__ __attribute__((always_inline)) __attribute__((device)) __attribute__((host))\n"
    "    T operator()(T const& scalar) const {\n"
    "      ElementwiseUnaryOp<T> op;\n"
    "\n"
    "      return op(scalar);\n"
    "    }\n"
    "\n"
    "    using Params =\n"
    "        cutlass::epilogue::thread::LinearCombinationGenericParams<T>;\n"
    "\n"
    "    __inline__ __attribute__((always_inline)) __attribute__((device)) __attribute__((host))\n"
    "    T operator()(T const& scalar, Params const& params_) const {\n"
    "      return this->operator()(scalar);\n"
    "    }\n"
    "  };\n"
    "\n"
    "  template <typename T, int N>\n"
    "  struct EpilogueFunctor<cutlass::Array<T, N>> {\n"
    "    static const bool kIsHeavy = EpilogueIsHeavy;\n"
    "\n"
    "    __inline__ __attribute__((always_inline)) __attribute__((device)) __attribute__((host))\n"
    "    cutlass::Array<T, N> operator()(cutlass::Array<T, N>\n"
    "                                    const& frag) const {\n"
    "      cutlass::Array<T, N> result;\n"
    "      ElementwiseUnaryOp<T> op;\n"
    "\n"
    "      #pragma unroll\n"
    "      for (int i = 0; i < N; ++i) {\n"
    "        result[i] = op(frag[i]);\n"
    "      }\n"
    "\n"
    "      return result;\n"
    "    }\n"
    "\n"
    "    using Params =\n"
    "        cutlass::epilogue::thread::LinearCombinationGenericParams<T>;\n"
    "\n"
    "    __inline__ __attribute__((always_inline)) __attribute__((device)) __attribute__((host))\n"
    "    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& frag,\n"
    "                                    Params const& params_) const {\n"
    "      return this->operator()(frag);\n"
    "    }\n"
    "  };\n"
    "\n"
    " public:\n"
    "  SpecializedGemmFusion(void* stream, int64_t batch_size, int64_t m,\n"
    "                        int64_t n, int64_t k, const void* A,\n"
    "                        const void* B, void* D)\n"
    "      : stream_(stream),\n"
    "        batch_size_(batch_size),\n"
    "        m_(m),\n"
    "        n_(n),\n"
    "        k_(k),\n"
    "        A_(A),\n"
    "        B_(B),\n"
    "        D_(D) {}\n"
    "  bool run();\n"
    "\n"
    " private:\n"
    "  void* stream_;\n"
    "  int64_t batch_size_;\n"
    "  int64_t m_;\n"
    "  int64_t n_;\n"
    "  int64_t k_;\n"
    "  const void* A_;\n"
    "  const void* B_;\n"
    "  void* D_;\n"
    "};\n"
    "\n"
    "template <>\n"
    "struct SpecializedGemmFusion::ElementwiseUnaryOp<EpilogueElementType> {\n"
    "  __inline__ __attribute__((always_inline)) __attribute__((device)) __attribute__((host))\n"
    "  EpilogueElementType operator()(\n"
    "      EpilogueElementType const& input) const {\n"
    "SpecializedEpilogue\n"
    "  }\n"
    "};\n"
    "\n"
    "bool SpecializedGemmFusion::run() {\n"
    "  constexpr cutlass::FloatRoundStyle Round =\n"
    "      cutlass::FloatRoundStyle::round_to_nearest;\n"
    "  constexpr cutlass::ComplexTransform TransformA =\n"
    "      cutlass::ComplexTransform::kNone;\n"
    "  constexpr cutlass::ComplexTransform TransformB =\n"
    "      cutlass::ComplexTransform::kNone;\n"
    "\n"
    "  using ElementA = ElementAType;\n"
    "  using LayoutA = ElementALayout;\n"
    "  using ElementB = ElementBType;\n"
    "  using LayoutB = ElementBLayout;\n"
    "  using ElementOutput = ElementOutputType;\n"
    "  using LayoutOutput = ElementOutputLayout;\n"
    "  using ElementAccumulator = ElementAccumulatorType;\n"
    "  using OperatorClass = OperatorClassType;\n"
    "  using ArchTag = SMArch;\n"
    "\n"
    "  constexpr cutlass::epilogue::thread::ScaleType::Kind Scale =\n"
    "      EpilogueScaleKind;\n"
    "  constexpr bool IsHeavy = EpilogueIsHeavy;\n"
    "  constexpr int Count = EpilogueCountVectorized;\n"
    "  using ElementComputeEpilogue = EpilogueElementType;\n"
    "  using EpilogueOutputOp =\n"
    "      cutlass::epilogue::thread::LinearCombinationGeneric<\n"
    "          EpilogueFunctor, ElementOutput, Count, ElementAccumulator,\n"
    "          ElementComputeEpilogue, Scale, Round, IsHeavy>;\n"
    "\n"
    "  using GemmConfiguration =\n"
    "      cutlass::gemm::device::DefaultGemmConfiguration<\n"
    "          OperatorClass, ArchTag, ElementA, ElementB, ElementOutput,\n"
    "          ElementAccumulator>;\n"
    "  constexpr int AlignmentA = GemmConfiguration::kAlignmentA;\n"
    "  constexpr int AlignmentB = GemmConfiguration::kAlignmentB;\n"
    "  using Operator = GemmConfiguration::Operator;\n"
    "\n"
    "  constexpr bool GatherA = IsGatherA;\n"
    "  constexpr bool GatherB = IsGatherB;\n"
    "  constexpr bool ScatterD = IsScatterD;\n"
    "  using PermuteDLayout = EpiloguePermuteDLayout;\n"
    "\n"
    "  cutlass::gemm::GemmUniversalMode mode;\n"
    "  if (batch_size_ > 1) {\n"
    "    mode = cutlass::gemm::GemmUniversalMode::kBatched;\n"
    "  } else {\n"
    "    if (true) {\n"
    "      mode = cutlass::gemm::GemmUniversalMode::kGemm;\n"
    "    }\n"
    "  }\n"
    "\n"
    "  cutlass::gemm::GemmCoord problem_size(m_, n_, k_);\n"
    "\n"
    "  typename EpilogueOutputOp::Params epilogue(\n"
    "      ElementComputeEpilogue(1), ElementComputeEpilogue(0));\n"
    "\n"
    "  long long int batch_stride_A =\n"
    "      static_cast<long long int>(m_) * static_cast<long long int>(k_);\n"
    "  long long int batch_stride_B =\n"
    "      static_cast<long long int>(k_) * static_cast<long long int>(n_);\n"
    "  long long int batch_stride_C = 0;\n"
    "  long long int batch_stride_D =\n"
    "      static_cast<long long int>(m_) * static_cast<long long int>(n_);\n"
    "\n"
    "  int const lda = std::is_same<LayoutA, cutlass::layout::ColumnMajor>::value ? m_ : k_;\n"
    "  int const ldb = std::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? k_ : n_;\n"
    "  int const ldc = 0;\n"
    "  int const ldd = n_;\n"
    "\n"
    "  int const* ptr_gather_A_indices = nullptr;\n"
    "  int const* ptr_gather_B_indices = nullptr;\n"
    "  int const* ptr_scatter_D_indices = nullptr;\n"
    "\n"
    "\n"
    "  if (true) {\n"
    "    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;\n"
    "    using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;\n"
    "    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;\n"
    "    using ThreadblockSwizzle =\n"
    "        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;\n"
    "    constexpr int Stages = 3;\n"
    "\n"
    "    using Gemm = cutlass::gemm::device::GemmUniversal<\n"
    "        ElementA, LayoutA, ElementB, LayoutB, ElementOutput,\n"
    "        LayoutOutput, ElementAccumulator, OperatorClass, ArchTag,\n"
    "        ThreadblockShape, WarpShape, InstructionShape,\n"
    "        EpilogueOutputOp, ThreadblockSwizzle, Stages, AlignmentA,\n"
    "        AlignmentB, Operator, TransformA, TransformB, GatherA, GatherB,\n"
    "        ScatterD, PermuteDLayout>;\n"
    "\n"
    "    typename Gemm::Arguments arguments{mode,\n"
    "                                       problem_size,\n"
    "                                       static_cast<int>(batch_size_),\n"
    "                                       epilogue,\n"
    "                                       A_,\n"
    "                                       B_,\n"
    "                                       nullptr,\n"
    "                                       D_,\n"
    "                                       batch_stride_A,\n"
    "                                       batch_stride_B,\n"
    "                                       batch_stride_C,\n"
    "                                       batch_stride_D,\n"
    "                                       lda,\n"
    "                                       ldb,\n"
    "                                       ldc,\n"
    "                                       ldd,\n"
    "                                       ptr_gather_A_indices,\n"
    "                                       ptr_gather_B_indices,\n"
    "                                       ptr_scatter_D_indices};\n"
    "\n"
    "    size_t workspace_size = Gemm::get_workspace_size(arguments);\n"
    "    cutlass::device_memory::allocation<uint8_t>\n"
    "        workspace(workspace_size);\n"
    "\n"
    "    Gemm gemm_op;\n"
    "    cutlass::Status status = gemm_op.can_implement(arguments);\n"
    "    if (status != cutlass::Status::kSuccess) {\n"
    "      return false;\n"
    "    }\n"
    "\n"
    "    cudaStream_t stream = static_cast<cudaStream_t>(stream_);\n"
    "    status = gemm_op.initialize(arguments, workspace.get(), stream);\n"
    "    if (status != cutlass::Status::kSuccess) {\n"
    "      return false;\n"
    "    }\n"
    "    if (true) {"
    "      ElementAType* A_host = (ElementAType*)malloc(sizeof(ElementAType) * batch_size_ * m_ * k_);\n"
    "      ElementAType* B_host = (ElementBType*)malloc(sizeof(ElementBType) * batch_size_ * k_ * n_);\n"
    "      cudaMemcpy(A_host, A_, sizeof(ElementAType) * batch_size_ * m_ * k_, cudaMemcpyDefault);\n"
    "      cudaMemcpy(B_host, B_, sizeof(ElementBType) * batch_size_ * k_ * n_, cudaMemcpyDefault);\n"
    "      for (int b = 0; b < 2; b++) {\n"
    "        for (int m = 0; m < 4; m++) {\n"
    "          for (int k = 0; k < 4; k++) {\n"
    "            std::cout << \"A val at \" << b << \",\" << m << \",\" << k << \": \"\n"
    "                      << A_host[b * m_ * k_ + m * k_ + k] << std::endl;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "      for (int b = 0; b < 2; b++) {\n"
    "        for (int k = 0; k < 4; k++) {\n"
    "          for (int n = 0; n < 4; n++) {\n"
    "            std::cout << \"B val at \" << b << \",\" << k << \",\" << n <<\": \"\n"
    "                      << B_host[b * k_ * n_ + k * n_ + n] << std::endl;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "\n"
    "    status = gemm_op();\n"
    "    if (status != cutlass::Status::kSuccess) {\n"
    "      return false;\n"
    "    }\n"
    "  }\n"
    "  return true;\n"
    "}\n"
    "\n"
    "extern \"C\"\n"
    "bool gemmFusionFunc(void* stream, void** params) {\n"
    "\n"
    "  int64_t size_struct = 3 + GRank * 2;\n"
    "  int64_t batch_size = 1;\n"
    "  for (int64_t i = 0; i < GRank - 2; ++i) {\n"
    "    int64_t* a_size = reinterpret_cast<int64_t*>(params[3 + i]);\n"
    "    batch_size *= *a_size;\n"
    "  }\n"
    "  if (true) {\n"
    "    std::cout << \"[ZZ] batch_size: \" << batch_size << std::endl;\n"
    "    std::cout << \"[ZZ] A sizes.\\n\";\n"
    "    for (int i = 0; i < GRank; i++) {\n"
    "      int64_t* a_size = reinterpret_cast<int64_t*>(params[3 + i]);\n"
    "      std::cout << i << \":\" << *a_size << \"\\n\";\n"
    "    }\n"
    "    std::cout << \"[ZZ] B sizes.\\n\";\n"
    "    for (int i = 0; i < GRank; i++) {\n"
    "      int64_t* b_size = reinterpret_cast<int64_t*>(params[3 + size_struct + i]);\n"
    "      std::cout << i << \":\" << *b_size << \"\\n\";\n"
    "    }\n"
    "    std::cout << std::flush;\n"
    "  }\n"
    "\n"
    "  int64_t m = std::is_same<ElementALayout, cutlass::layout::ColumnMajor>::value ?\n"
    "      *reinterpret_cast<int64_t*>(params[3 + GRank - 1]) :\n"
    "      *reinterpret_cast<int64_t*>(params[3 + GRank - 2]);\n"
    "  int64_t k = std::is_same<ElementALayout, cutlass::layout::ColumnMajor>::value ?\n"
    "      *reinterpret_cast<int64_t*>(params[3 + GRank - 2]) :\n"
    "      *reinterpret_cast<int64_t*>(params[3 + GRank - 1]);\n"
    "  int64_t n = std::is_same<ElementBLayout, cutlass::layout::ColumnMajor>::value ?\n"
    "      *reinterpret_cast<int64_t*>(params[3 + size_struct + GRank - 2]) :\n"
    "      *reinterpret_cast<int64_t*>(params[3 + size_struct + GRank - 1]);\n"
    "\n"
    "  ElementAType* a = *reinterpret_cast<ElementAType**>(params[1]);\n"
    "  ElementBType* b = *reinterpret_cast<ElementBType**>(params[1 + size_struct]);\n"
    "  ElementOutputType* d =\n"
    "      *reinterpret_cast<ElementOutputType**>(params[1 + 2 * size_struct]);\n"
    "\n"
    "  SpecializedGemmFusion specialization(stream, batch_size, m, n, k, a, b, d);\n"
    "  return specialization.run();\n"
    "}";
// clang-format on

int64_t stringReplaceInplace(std::string& subject, const std::string& oldsub,
                             const std::string& newsub, bool replace_all) {
  if (oldsub.empty()) {
    return 0;
  }
  int64_t count = 0;
  size_t pos = 0;
  while ((pos = subject.find(oldsub, pos)) != std::string::npos) {
    subject.replace(pos, oldsub.size(), newsub);
    count++;
    pos += newsub.size();
    if (!replace_all) {
      break;
    }
  }
  return count;
}

struct DiscCompIntensFusionToCUDASourcePass
    : public DiscCompIntensFusionToCUDASourcePassBase<
          DiscCompIntensFusionToCUDASourcePass> {
 public:
  DiscCompIntensFusionToCUDASourcePass() = delete;
  explicit DiscCompIntensFusionToCUDASourcePass(int cc_major, int cc_minor) {
    cc_major_ = cc_major;
    cc_minor_ = cc_minor;
  }
  void runOnOperation() override;

 private:
  int cc_major_;
  int cc_minor_;

 private:
  bool generateCUDASourceForCompIntensFusionFunc(func::FuncOp func);

  bool isHeavyEpilogue(func::FuncOp func);
  bool getCUDATypeString(Type type, std::string& type_str);
  bool getLayoutStrings(const mhlo::DotDimensionNumbersAttr& dimension_numbers,
                        SmallVector<std::string>& layout);
  bool getAccumulatorTypeString(Type output_type,
                                std::string& accumulator_type_str);
  bool getOperatorClassTypeString(std::string& operator_class_type);
  bool getSMArchString(std::string& sm_arch);
  bool getScaleKindString(std::string& scale_kind);
  bool getCountVectorizedString(const std::string& element_output_type,
                                std::string& count_vectorized);
  bool isGatherA(func::FuncOp func);
  bool isGatherB(func::FuncOp func);
  bool isScatterD(func::FuncOp func);
  bool getPermuteDLayoutString(func::FuncOp func,
                               std::string& permute_d_layout);

  void getResults(func::FuncOp func, SmallVector<Value>& results);
  void getEffectiveOperands(func::FuncOp func, SmallVector<Value>& operands);
};

bool DiscCompIntensFusionToCUDASourcePass::isHeavyEpilogue(func::FuncOp func) {
  // TODO: check whether it is heavy or not according to the logic in CUTLASS.
  // return false;
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getCUDATypeString(
    Type type, std::string& type_str) {
  if (type.isInteger(1)) {
    type_str = "cutlass::uint1b_t";
  } else if (type.isSignedInteger(8)) {
    type_str = "cutlass::int8b_t";
  } else if (type.isUnsignedInteger(8)) {
    type_str = "cutlass::uint8b_t";
  } else if (type.isSignedInteger(32)) {
    type_str = "cutlass::int32b_t";
  } else if (type.isUnsignedInteger(32)) {
    type_str = "cutlass::uint32b_t";
  } else if (type.isBF16()) {
    type_str = "cutlass::bfloat16_t";
  } else if (type.isF16()) {
    type_str = "cutlass::half_t";
  } else if (type.isF32()) {
    type_str = "float";
  } else if (type.isF64()) {
    type_str = "double";
  } else {
    return false;
  }
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getLayoutStrings(
    const mhlo::DotDimensionNumbersAttr& dimension_numbers,
    SmallVector<std::string>& layout) {
  auto lhs_contracting_dims = dimension_numbers.getLhsContractingDimensions();
  auto rhs_contracting_dims = dimension_numbers.getRhsContractingDimensions();
  // Only deal with dot with 1 contracting dim.
  if (lhs_contracting_dims.size() != 1 || rhs_contracting_dims.size() != 1) {
    return false;
  }

  auto lhs_batching_dims = dimension_numbers.getLhsBatchingDimensions();
  auto rhs_batching_dims = dimension_numbers.getRhsBatchingDimensions();
  for (int64_t i = 0; i < lhs_batching_dims.size(); i++) {
    // Only deal with dot whose batch dims are the significant dims.
    if (lhs_batching_dims[i] > lhs_batching_dims.size() ||
        rhs_batching_dims[i] > rhs_batching_dims.size()) {
      return false;
    }
  }
  if (lhs_contracting_dims[0] == lhs_batching_dims.size() + 1) {
    layout.emplace_back("cutlass::layout::RowMajor");
  } else {
    layout.emplace_back("cutlass::layout::ColumnMajor");
  }

  if (rhs_contracting_dims[0] == rhs_batching_dims.size()) {
    layout.emplace_back("cutlass::layout::RowMajor");
  } else {
    layout.emplace_back("cutlass::layout::ColumnMajor");
  }

  // According to https://www.tensorflow.org/xla/operation_semantics#dotgeneral,
  // the result layout follows the order of [batch0, batch1, .., m, n].
  layout.emplace_back("cutlass::layout::RowMajor");

  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getAccumulatorTypeString(
    Type type, std::string& accumulator_type_str) {
  if (type.isSignedInteger() && type.getIntOrFloatBitWidth() <= 32) {
    accumulator_type_str = "cutlass::int32b_t";
  } else if (type.isUnsignedInteger() && type.getIntOrFloatBitWidth() <= 32) {
    accumulator_type_str = "cutlass::uint32b_t";
  } else if (type.isBF16() || type.isF16() || type.isF32()) {
    accumulator_type_str = "float";
  } else if (type.isF64()) {
    accumulator_type_str = "double";
  } else {
    return false;
  }
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getOperatorClassTypeString(
    std::string& operator_class_type) {
  // Currently we only support tensor op.
  operator_class_type = "cutlass::arch::OpClassTensorOp";
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getSMArchString(
    std::string& sm_arch) {
  if (cc_major_ == 8) {
    sm_arch = "cutlass::arch::Sm80";
  } else if (cc_major_ == 7) {
    if (cc_minor_ == 5) {
      sm_arch = "cutlass::arch::Sm75";
    } else {
      sm_arch = "cutlass::arch::Sm70";
    }
  } else {
    return false;
  }
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getScaleKindString(
    std::string& scale_kind) {
  // TODO: update according to the problem.
  // scale_kind = "cutlass::epilogue::thread::ScaleType::NoBetaScaling";
  scale_kind = "cutlass::epilogue::thread::ScaleType::Nothing";
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getCountVectorizedString(
    const std::string& element_output_type, std::string& count_vectorized) {
  count_vectorized =
      "128 / cutlass::sizeof_bits<" + element_output_type + ">::value";
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::isGatherA(func::FuncOp func) {
  // TODO: update according to the problem.
  return false;
}

bool DiscCompIntensFusionToCUDASourcePass::isGatherB(func::FuncOp func) {
  // TODO: update according to the problem.
  return false;
}

bool DiscCompIntensFusionToCUDASourcePass::isScatterD(func::FuncOp func) {
  // TODO: update according to the problem.
  return false;
}

bool DiscCompIntensFusionToCUDASourcePass::getPermuteDLayoutString(
    func::FuncOp func, std::string& permute_d_layout) {
  // TODO: update according to the problem.
  permute_d_layout = "cutlass::layout::NoPermute";
  return true;
}

void DiscCompIntensFusionToCUDASourcePass::getResults(
    func::FuncOp func, SmallVector<Value>& results) {
  DenseSet<Operation*> op_set;
  auto ops = func.getRegion().getOps();
  for (auto& op : func.getRegion().getOps()) {
    op_set.insert(&op);
  }

  results.clear();
  func.walk([&](Operation* op) {
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value v : op->getOperands().drop_front(num_input_operand)) {
      bool has_internal_user = false;
      for (Operation* user : getValueUsers(v)) {
        if (op == user) {
          continue;
        }
        has_internal_user |= op_set.contains(user);
      }
      if (!has_internal_user) {
        results.push_back(v);
      }
    }
  });
}

void DiscCompIntensFusionToCUDASourcePass::getEffectiveOperands(
    func::FuncOp func, SmallVector<Value>& operands) {
  DenseSet<Operation*> op_set;
  auto ops = func.getRegion().getOps();
  for (auto& op : func.getRegion().getOps()) {
    op_set.insert(&op);
  }

  operands.clear();
  func.walk([&](Operation* op) {
    SmallVector<Value> ins;
    if (isa<lmhlo::ConstantOp>(op)) {
      // The constant op is splat constant, which should be checked in the
      // fusion pass.
      return;
    } else {
      // The broadcast-in-dim reads constant op, which should be check in the
      // fusion pass.
      int num_input_operand =
          isa<lmhlo::DynamicBroadcastInDimOp>(op)
              ? 1
              : op->getNumOperands() - getNumResultOperands(op);
      for (Value v : op->getOperands().take_front(num_input_operand)) {
        bool has_internal_writter = false;
        for (Operation* user : getValueUsers(v)) {
          if (op == user) {
            continue;
          }
          if (op_set.contains(user) && IsOpWriteValue(user, v)) {
            has_internal_writter = true;
            break;
          }
        }
        if (!has_internal_writter) {
          operands.emplace_back(v);
        }
      }
    }
  });
}

bool DiscCompIntensFusionToCUDASourcePass::
    generateCUDASourceForCompIntensFusionFunc(func::FuncOp func) {
  std::string cuda_code = gemm_fusion_template;

  // Values to obtain according to func.
  std::string specialized_class_name;
  std::string specialized_epilogue;
  std::string epilogue_is_heavy;
  std::string gemm_rank;
  std::string element_a_type;
  std::string element_a_layout;
  std::string element_b_type;
  std::string element_b_layout;
  std::string element_output_type;
  std::string element_output_layout;
  std::string element_accumulator_type;
  std::string operator_class_type;
  std::string sm_arch;
  std::string scale_kind;
  std::string count_vectorized;
  std::string epilogue_type;
  std::string gather_a;
  std::string gather_b;
  std::string scatter_d;
  std::string permute_d_layout;
  std::string gemm_fusion_func_name;

  gemm_fusion_func_name = func.getName();
  specialized_class_name = gemm_fusion_func_name + "_specialized";

  specialized_epilogue = "";

  epilogue_is_heavy = isHeavyEpilogue(func) ? "true" : "false";

  SmallVector<Operation*> gemms;
  func->walk([&](lmhlo::DotGeneralOp op) { gemms.push_back(op); });
  // Currently, we only deal with the fusion with a single GEMM.
  assert(gemms.size() == 1 && "GEMM number in kDot fusion is not 1.");
  lmhlo::DotGeneralOp dot = dyn_cast<lmhlo::DotGeneralOp>(gemms[0]);
  Value A = dot->getOperand(0);
  Value B = dot->getOperand(1);
  Value D = dot->getOperand(2);
  auto a_type = A.getType().dyn_cast<MemRefType>();
  auto b_type = B.getType().dyn_cast<MemRefType>();
  auto d_type = D.getType().dyn_cast<MemRefType>();

  gemm_rank = std::to_string(a_type.getRank());

  if (!getCUDATypeString(a_type.getElementType(), element_a_type) ||
      !getCUDATypeString(b_type.getElementType(), element_b_type) ||
      !getCUDATypeString(d_type.getElementType(), element_output_type)) {
    return false;
  }
  SmallVector<std::string> layouts;
  if (!getLayoutStrings(dot.getDotDimensionNumbers(), layouts)) {
    return false;
  }
  element_a_layout = layouts[0];
  element_b_layout = layouts[1];
  element_output_layout = layouts[2];

  if (!getAccumulatorTypeString(d_type.getElementType(),
                                element_accumulator_type) ||
      !getOperatorClassTypeString(operator_class_type) ||
      !getSMArchString(sm_arch) || !getScaleKindString(scale_kind) ||
      !getCountVectorizedString(element_output_type, count_vectorized) ||
      !getPermuteDLayoutString(func, permute_d_layout)) {
    return false;
  }

  // TODO: support more epilogue types.
  epilogue_type = element_output_type;
  gather_a = isGatherA(func) ? "true" : "false";
  gather_b = isGatherB(func) ? "true" : "false";
  scatter_d = isScatterD(func) ? "true" : "false";

  std::string intent = "    ";
  SourceEmitterCUDA source_emitter;
  SourceEmitterCUDA::ValueNameBinding binding;
  for (auto& op : func.getRegion().getOps()) {
    if (isa<lmhlo::DotGeneralOp>(&op)) {
      source_emitter.bindValueNames(SmallVector<Value>({op.getOperand(2)}),
                                    SmallVector<std::string>({"input"}),
                                    binding);
    } else if (!isa<func::ReturnOp>(&op)) {
      assert(source_emitter.isSupportedOp(&op) && "Encounter unsupported op.");
      auto instruction = source_emitter.EmitOp(&op, binding);
      if (!instruction.hasValue()) {
        return false;
      } else {
        specialized_epilogue += intent + instruction.value() + ";\n";
      }
    }
  }
  // Append return instruction.
  SmallVector<Value> results;
  getResults(func, results);
  // Only support one result currently.
  if (results.size() != 1) {
    return false;
  }
  std::string result_name = binding[results[0]];
  specialized_epilogue += intent + "return " + result_name + ";";

  // Replace newly generated code in the template.
  stringReplaceInplace(cuda_code, kSpecializedClassName, specialized_class_name,
                       true);
  stringReplaceInplace(cuda_code, kSpecializedEpilogue, specialized_epilogue,
                       true);
  stringReplaceInplace(cuda_code, kEpilogueIsHeavy, epilogue_is_heavy, true);
  stringReplaceInplace(cuda_code, kGRank, gemm_rank, true);
  stringReplaceInplace(cuda_code, kElementAType, element_a_type, true);
  stringReplaceInplace(cuda_code, kElementALayout, element_a_layout, true);
  stringReplaceInplace(cuda_code, kElementBType, element_b_type, true);
  stringReplaceInplace(cuda_code, kElementBLayout, element_b_layout, true);
  stringReplaceInplace(cuda_code, kElementOutputType, element_output_type,
                       true);
  stringReplaceInplace(cuda_code, kElementOutputLayout, element_output_layout,
                       true);
  stringReplaceInplace(cuda_code, kElementAccumulatorType,
                       element_accumulator_type, true);
  stringReplaceInplace(cuda_code, kOperatorClassType, operator_class_type,
                       true);
  stringReplaceInplace(cuda_code, kSMArch, sm_arch, true);
  stringReplaceInplace(cuda_code, kScaleKind, scale_kind, true);
  stringReplaceInplace(cuda_code, kCountVectorized, count_vectorized, true);
  stringReplaceInplace(cuda_code, kEpilogueType, epilogue_type, true);
  stringReplaceInplace(cuda_code, kGatherA, gather_a, true);
  stringReplaceInplace(cuda_code, kGatherB, gather_b, true);
  stringReplaceInplace(cuda_code, kScatterD, scatter_d, true);
  stringReplaceInplace(cuda_code, kPermuteDLayout, permute_d_layout, true);
  stringReplaceInplace(cuda_code, kGemmFusionFuncName, gemm_fusion_func_name,
                       true);

#if 0
  llvm::errs() << "[ZZ] generated cuda code: " << cuda_code << "\n";
#endif

  OpBuilder builder(func);
  Location loc = func->getLoc();

  // Create source code op containing the source code string.
  SmallVector<Value> operands;
  getEffectiveOperands(func, operands);

  // Replace fun and it's calls with `SourceCodeOp`.
  SmallVector<int32_t> effective_operand_pos;
  SmallVector<int32_t> effective_result_pos;
  for (auto argument : llvm::enumerate(func.getArguments())) {
    if (llvm::find(operands, argument.value()) != operands.end()) {
      effective_operand_pos.push_back(argument.index());
    }
    if (llvm::find(results, argument.value()) != results.end()) {
      effective_result_pos.push_back(argument.index());
    }
  }

  auto module_op = func->getParentOfType<ModuleOp>();
  Optional<SymbolTable::UseRange> symbol_uses = func.getSymbolUses(module_op);
  for (SymbolTable::SymbolUse symbol_use : *symbol_uses) {
    Operation* user = symbol_use.getUser();
    auto call = dyn_cast<func::CallOp>(user);
    if (!call) {
      continue;
    }
    SmallVector<Value> operands;
    for (auto idx : effective_operand_pos) {
      operands.push_back(call.getOperand(idx));
    }
    SmallVector<Value> results;
    for (auto idx : effective_result_pos) {
      results.push_back(call.getOperand(idx));
    }
    OpBuilder builder_call(user);
    auto source_code_op = builder_call.create<lmhlo_disc::SourceCodeOp>(
        call->getLoc(), llvm::None, operands, results, cuda_code,
        gemm_fusion_func_name);

    assert(user->getUsers().empty() &&
           "Call of gemm fusion should have no users.");
    user->erase();
  }
  func->erase();

  return true;
}

void DiscCompIntensFusionToCUDASourcePass::runOnOperation() {
  ModuleOp module_op = getOperation();

  SmallVector<func::FuncOp> comp_intens_fusions;
  module_op->walk([&](func::FuncOp func) {
    if (func->getAttrOfType<StringAttr>(kFuncCompIntensFusionAttr)) {
      comp_intens_fusions.push_back(func);
    }
  });

  for (auto func : comp_intens_fusions) {
    generateCUDASourceForCompIntensFusionFunc(func);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createDiscCompIntensFusionToCUDASourcePass(int cc_major, int cc_minor) {
  return std::make_unique<DiscCompIntensFusionToCUDASourcePass>(cc_major,
                                                                cc_minor);
}

}  // namespace disc_ral
}  // namespace mlir
