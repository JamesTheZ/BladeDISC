#include <algorithm>
#include <iostream>

#include "cutlass/array.h"
#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

// For GEMM

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;  // data type of epilogue

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using OperatorClass = cutlass::arch::OpClassTensorOp;

constexpr bool GatherA = false;
constexpr bool GatherB = false;
constexpr bool ScatterD = false;

using PermuteDLayout = cutlass::layout::NoPermute;

// For epilogue.

constexpr cutlass::epilogue::thread::ScaleType::Kind Scale =
    cutlass::epilogue::thread::ScaleType::NoBetaScaling;

constexpr cutlass::FloatRoundStyle Round =
    cutlass::FloatRoundStyle::round_to_nearest;

constexpr bool IsHeavy = false;

template <typename T>
struct ElementwiseUnaryOp {
  CUTLASS_HOST_DEVICE
  T operator()(T const& input) const {
    //////////// To update according to the problem ////////////
    return input + 1;
    ////////////////////////////////////////////////////////////
  }
};

// TBD: specialized ElementwiseUnaryOp

template <typename T>
struct EpilogueFunctor {
  //////////// To update according to the problem ////////////
  static const bool kIsHeavy = false;
  ////////////////////////////////////////////////////////////

  CUTLASS_HOST_DEVICE
  T operator()(T const& scalar) const {
    ElementwiseUnaryOp<T> op;

    return op(scalar);
  }

  using Params = cutlass::epilogue::thread::LinearCombinationGenericParams<T>;

  CUTLASS_HOST_DEVICE
  T operator()(T const& scalar, Params const& params_) const {
    return this->operator()(scalar);
  }
};

template <typename T, int N>
struct EpilogueFunctor<cutlass::Array<T, N>> {
  //////////// To update according to the problem ////////////
  static const bool kIsHeavy = false;
  ////////////////////////////////////////////////////////////

  CUTLASS_HOST_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& frag) const {
    cutlass::Array<T, N> result;
    ElementwiseUnaryOp<T> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(frag[i]);
    }

    return result;
  }

  using Params = cutlass::epilogue::thread::LinearCombinationGenericParams<T>;

  CUTLASS_HOST_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& frag,
                                  Params const& params_) const {
    return this->operator()(frag);
  }
};

/////////////////////////////////////////////////////////////

///////////// To update according to platform. //////////////

using ArchTag = cutlass::arch::Sm80;

/////////////////////////////////////////////////////////////

///////////// To update according to profiling. /////////////

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using ThreadblockSwizzle =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
// cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

constexpr int Stages = 2;

// Number of elements computed per operation, i.e., per vectorized memory
// access. Usually it is 128/sizeof_bits<ElementOutput_>, but we use 64 or 32
// sometimes when there are not enough data to store.
constexpr int Count = 128 / cutlass::sizeof_bits<ElementOutput>::value;

/////////////////////////////////////////////////////////////

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGeneric<
    EpilogueFunctor, ElementOutput, Count, ElementAccumulator,
    ElementComputeEpilogue, Scale, Round, IsHeavy>;

using GemmConfiguration = cutlass::gemm::device::DefaultGemmConfiguration<
    OperatorClass, ArchTag, ElementA, ElementB, ElementOutput,
    ElementAccumulator>;

constexpr int AlignmentA = GemmConfiguration::kAlignmentA;
constexpr int AlignmentB = GemmConfiguration::kAlignmentB;

using Operator = GemmConfiguration::Operator;

constexpr cutlass::ComplexTransform TransformA =
    cutlass::ComplexTransform::kNone;
constexpr cutlass::ComplexTransform TransformB =
    cutlass::ComplexTransform::kNone;

using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutOutput,
    ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
    InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages, AlignmentA,
    AlignmentB, Operator, TransformA, TransformB, GatherA, GatherB, ScatterD,
    PermuteDLayout>;

int main() {
  ///////////// To update according to problem. ///////////////
  const int length_m = 128;
  const int length_n = 128;
  const int length_k = 128;

  cutlass::gemm::GemmUniversalMode mode =
      cutlass::gemm::GemmUniversalMode::kBatched;
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  const int batch_count = 2;
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
  typename EpilogueOutputOp::Params epilogue(alpha, beta);

  cutlass::HostTensor<ElementA, LayoutA> tensor_a(
      {length_m * batch_count, length_k});
  cutlass::HostTensor<ElementB, LayoutB> tensor_b(
      {length_k * batch_count, length_n});
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      {length_m * batch_count, length_n});

  // Fill input and output matrices on host using CUTLASS helper functions
  for (int b = 0; b < batch_count; b++) {
    for (int k = 0; k < length_k; k++) {
      for (int m = 0; m < length_m; m++) {
        tensor_a.host_view().at({b * length_m + m, k}) = b * 1000 + m;
      }
    }
  }
  // Fill identity for each gemm in the batch for B.
  for (int n = 0; n < length_n; n++) {
    for (int b = 0; b < batch_count; b++) {
      for (int k = 0; k < length_k; k++) {
        tensor_b.host_view().at({b * length_k + k, n}) = (k == n ? 1 : 0);
      }
    }
  }
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // fill with zeros

  int const lda = length_k;
  int const ldb = length_n;
  int const ldc = 0;
  int const ldd = length_n;

  void const* ptr_A = tensor_a.device_data();
  void const* ptr_B = tensor_b.device_data();
  void const* ptr_C = nullptr;
  void* ptr_D = tensor_d.device_data();

  long long int batch_stride_A =
      static_cast<long long int>(lda) * static_cast<long long int>(length_m);
  long long int batch_stride_B =
      static_cast<long long int>(ldb) * static_cast<long long int>(length_k);
  long long int batch_stride_C = 0;
  long long int batch_stride_D =
      static_cast<long long int>(ldd) * static_cast<long long int>(length_m);

  int const* ptr_gather_A_indices = nullptr;
  int const* ptr_gather_B_indices = nullptr;
  int const* ptr_scatter_D_indices = nullptr;

  /////////////////////////////////////////////////////////////

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_d.sync_device();

  typename Gemm::Arguments arguments{mode,
                                     problem_size,
                                     batch_count,
                                     epilogue,
                                     ptr_A,
                                     ptr_B,
                                     ptr_C,
                                     ptr_D,
                                     batch_stride_A,
                                     batch_stride_B,
                                     batch_stride_C,
                                     batch_stride_D,
                                     lda,
                                     ldb,
                                     ldc,
                                     ldd,
                                     ptr_gather_A_indices,
                                     ptr_gather_B_indices,
                                     ptr_scatter_D_indices};

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  status = gemm_op();
  CUTLASS_CHECK(status);

  auto error = cudaDeviceSynchronize();

#if 1
  tensor_d.sync_host();
  // print partial.
  if (true) {
    for (int b = 0; b < batch_count; b++) {
      std::cout << "[ZZ] mat " << b << std::endl;
      for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
          std::cout << tensor_d.host_view().at(
                           {row + b * length_m, col})  // row major
                    << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  // print all.
  if (false) {
    for (int row = 0; row < length_m; row++) {
      for (int b = 0; b < batch_count; b++) {
        for (int col = 0; col < length_n; col++) {
          std::cout << tensor_d.host_view().at({row, col + b * length_n})
                    << " ";
        }
      }
      std::cout << std::endl;
    }
  }
#endif

  return 0;
}