R"bladedisc_rstr(

class __SpecializedGemmFusion__ {
 public:
  template <typename T>
  struct ElementwiseUnaryOp {
    __inline__ __attribute__((always_inline)) __attribute__((device))
    T operator()(T const& input) const {
      return input;
    }
  };

  template <typename T>
  struct EpilogueFunctor {
    static const bool kIsHeavy = __EpilogueIsHeavy__;

    __inline__ __attribute__((always_inline)) __attribute__((device))
    T operator()(T const& scalar) const {
      ElementwiseUnaryOp<T> op;

      return op(scalar);
    }

    using Params =
        cutlass::epilogue::thread::LinearCombinationGenericParams<T>;

    __inline__ __attribute__((always_inline)) __attribute__((device))
    T operator()(T const& scalar, Params const& params_) const {
      return this->operator()(scalar);
    }
  };

  template <typename T, int N>
  struct EpilogueFunctor<cutlass::Array<T, N>> {
    static const bool kIsHeavy = __EpilogueIsHeavy__;

    __inline__ __attribute__((always_inline)) __attribute__((device))
    cutlass::Array<T, N> operator()(cutlass::Array<T, N>
                                    const& frag) const {
      cutlass::Array<T, N> result;
      ElementwiseUnaryOp<T> op;

      #pragma unroll
      for (int i = 0; i < N; ++i) {
        result[i] = op(frag[i]);
      }

      return result;
    }

    using Params =
        cutlass::epilogue::thread::LinearCombinationGenericParams<T>;

    __inline__ __attribute__((always_inline)) __attribute__((device))
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& frag,
                                    Params const& params_) const {
      return this->operator()(frag);
    }
  };

 public:
  __SpecializedGemmFusion__(void* stream, int64_t batch_size, int64_t m,
                        int64_t n, int64_t k, const void** ptr_A_array,
                        const void** ptr_B_array, void** ptr_D_array)
      : stream_(stream),
        batch_size_(batch_size),
        m_(m),
        n_(n),
        k_(k),
        ptr_A_array_(ptr_A_array),
        ptr_B_array_(ptr_B_array),
        ptr_D_array_(ptr_D_array) {}
  bool run();

 private:
  void* stream_;
  int64_t batch_size_;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  const void** ptr_A_array_;
  const void** ptr_B_array_;
  void** ptr_D_array_;
};

template <>
struct __SpecializedGemmFusion__::ElementwiseUnaryOp<__EpilogueElementType__> {
  __inline__ __attribute__((always_inline)) __attribute__((device))
  __EpilogueElementType__ operator()(
      __EpilogueElementType__ const& input) const {
__SpecializedEpilogue__
  }
};

bool __SpecializedGemmFusion__::run() {
  bool debug = false;
  if (debug) {
    __ElementAType__* A_host = (__ElementAType__*)malloc(sizeof(__ElementAType__) * batch_size_ * m_ * k_);
    __ElementAType__* B_host = (__ElementBType__*)malloc(sizeof(__ElementBType__) * batch_size_ * k_ * n_);
    for (int i = 0; i < batch_size_; i++) {
      int offset_A = i * m_ * k_;
      int offset_B = i * k_ * n_;
      cudaMemcpy(A_host + offset_A, ptr_A_array_[i], sizeof(__ElementAType__) * m_ * k_, cudaMemcpyDefault);
      cudaMemcpy(B_host + offset_B, ptr_B_array_[i], sizeof(__ElementBType__) * k_ * n_, cudaMemcpyDefault);
    }
    std::cout << "Some A value:" << std::endl;
    for (int b = 0; b < 1; b++) {
      for (int m = 0; m < 4; m++) {
        for (int k = 0; k < 4; k++) {
          std::cout << "A val at " << b << "," << m << "," << k << ": "
                    << A_host[b * m_ * k_ + m * k_ + k] << std::endl;
        }
      }
    }
    std::cout << "Some B value:" << std::endl;
    for (int b = 0; b < 1; b++) {
      for (int k = 0; k < 4; k++) {
        for (int n = 0; n < 4; n++) {
          std::cout << "B val at " << b << "," << k << "," << n <<": "
                    << B_host[b * k_ * n_ + k * n_ + n] << std::endl;
        }
      }
    }
  }

  constexpr cutlass::FloatRoundStyle Round =
      cutlass::FloatRoundStyle::round_to_nearest;
  constexpr cutlass::ComplexTransform TransformA =
      cutlass::ComplexTransform::kNone;
  constexpr cutlass::ComplexTransform TransformB =
      cutlass::ComplexTransform::kNone;

  using ElementA = __ElementAType__;
  using LayoutA = __ElementALayout__;
  using ElementB = __ElementBType__;
  using LayoutB = __ElementBLayout__;
  using ElementOutput = __ElementOutputType__;
  using LayoutOutput = __ElementOutputLayout__;
  using ElementAccumulator = __ElementAccumulatorType__;
  using OperatorClass = __OperatorClassType__;
  using ArchTag = __SMArch__;

  constexpr cutlass::epilogue::thread::ScaleType::Kind Scale =
      __EpilogueScaleKind__;
  constexpr bool IsHeavy = __EpilogueIsHeavy__;
  constexpr int Count = __EpilogueCountVectorized__;
  using ElementComputeEpilogue = __EpilogueElementType__;
  using EpilogueOutputOp =
      cutlass::epilogue::thread::LinearCombinationGeneric<
          EpilogueFunctor, ElementOutput, Count, ElementAccumulator,
          ElementComputeEpilogue, Scale, Round, IsHeavy>;

  constexpr bool GatherA = __IsGatherA__;
  constexpr bool GatherB = __IsGatherB__;
  constexpr bool ScatterD = __IsScatterD__;
  using PermuteDLayout = __EpiloguePermuteDLayout__;

  cutlass::gemm::GemmUniversalMode mode =
      cutlass::gemm::GemmUniversalMode::kArray;

  cutlass::gemm::GemmCoord problem_size(m_, n_, k_);

  typename EpilogueOutputOp::Params epilogue(
      ElementComputeEpilogue(1), ElementComputeEpilogue(0));

  long long int batch_stride_A = int64_t();
  long long int batch_stride_B = int64_t();
  long long int batch_stride_C = int64_t();
  long long int batch_stride_D = int64_t();

  int const lda = std::is_same<LayoutA, cutlass::layout::ColumnMajor>::value ? m_ : k_;
  int const ldb = std::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? k_ : n_;
  int const ldc = 0;
  int const ldd = std::is_same<LayoutOutput, cutlass::layout::ColumnMajor>::value ? m_ : n_;

  int const* ptr_gather_A_indices = nullptr;
  int const* ptr_gather_B_indices = nullptr;
  int const* ptr_scatter_D_indices = nullptr;

  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  int i_am_comment_the_following_rules_will_be_enriched;
  if (std::is_same<ElementA, cutlass::half_t>::value &&
      std::is_same<ElementB, cutlass::half_t>::value &&
      std::is_same<ElementOutput, cutlass::half_t>::value) {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using ThreadblockSwizzle =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
    constexpr int Stages = 3;

    using GemmConfiguration =
        cutlass::gemm::device::DefaultGemmConfiguration<
            cutlass::arch::OpClassTensorOp, ArchTag, cutlass::half_t, cutlass::half_t, cutlass::half_t,
            ElementAccumulator>;
    constexpr int AlignmentA = GemmConfiguration::kAlignmentA;
    constexpr int AlignmentB = GemmConfiguration::kAlignmentB;
    using Operator = GemmConfiguration::Operator;

    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, cutlass::half_t,
        LayoutOutput, ElementAccumulator, cutlass::arch::OpClassTensorOp, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, AlignmentA,
        AlignmentB, Operator, TransformA, TransformB, GatherA, GatherB,
        ScatterD, PermuteDLayout>;

    typename Gemm::Arguments arguments{mode,
                                       problem_size,
                                       static_cast<int>(batch_size_),
                                       epilogue,
                                       (void const *)ptr_A_array_,
                                       (void const *)ptr_B_array_,
                                       nullptr,
                                       (void       *)ptr_D_array_,
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
    cutlass::device_memory::allocation<uint8_t>
        workspace(workspace_size);

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return false;
    }
  } else if (std::is_same<ElementA, float>::value &&
      std::is_same<ElementB, float>::value &&
      std::is_same<ElementOutput, float>::value) {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    using ThreadblockSwizzle =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
    constexpr int Stages = 3;

    using GemmConfiguration =
        cutlass::gemm::device::DefaultGemmConfiguration<
            OperatorClass, ArchTag, float, float, float,
            ElementAccumulator>;
    constexpr int AlignmentA = GemmConfiguration::kAlignmentA;
    constexpr int AlignmentB = GemmConfiguration::kAlignmentB;
    using Operator = GemmConfiguration::Operator;

    using Gemm = cutlass::gemm::device::GemmUniversal<
        float, LayoutA, float, LayoutB, float,
        LayoutOutput, ElementAccumulator, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, AlignmentA,
        AlignmentB, Operator, TransformA, TransformB, GatherA, GatherB,
        ScatterD, PermuteDLayout>;

    typename Gemm::Arguments arguments{mode,
                                       problem_size,
                                       static_cast<int>(batch_size_),
                                       epilogue,
                                       (void const *)ptr_A_array_,
                                       (void const *)ptr_B_array_,
                                       nullptr,
                                       (void       *)ptr_D_array_,
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
    cutlass::device_memory::allocation<uint8_t>
        workspace(workspace_size);

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return false;
    }
  } else {
    return false;
  }

  if (debug) {
    cudaDeviceSynchronize();
    __ElementOutputType__* D_host = (__ElementOutputType__*)malloc(
        sizeof(__ElementOutputType__) * batch_size_ * m_ * n_);
    for (int i = 0; i < batch_size_; i++) {
      int offset_D = i * m_ * n_;
      cudaMemcpy(D_host + offset_D, ptr_D_array_[i], sizeof(__ElementOutputType__) * m_ * n_, cudaMemcpyDefault);
    }
    std::cout << "Some D value:" << std::endl;
    for (int b = 0; b < 1; b++) {
      for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
          std::cout << "D val at " << b << "," << m << "," << n << ": "
                    << D_host[b * m_ * n_ + m * n_ + n] << std::endl;
        }
      }
    }
  }

  return true;
}

extern "C"
bool __gemmFusionFunc__(void* stream, void** params) {

  int64_t size_struct = 3 + __GRank__ * 2;
  int64_t batch_size = __GEMM_NUM__;

  int param_permute[] = __ParameterPermute__;
  bool debug = false;
  if (debug) {
    std::cout << "param permute:" << param_permute[0] << ", "
              << param_permute[1] << ", " << param_permute[2] << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "A sizes." << std::endl;
    for (int i = 0; i < __GRank__; i++) {
      int64_t* a_size = reinterpret_cast<int64_t*>(
          params[3 + i + param_permute[0] * size_struct]);
      std::cout << i << ":" << *a_size << std::endl;
    }
    std::cout << "B sizes." << std::endl;
    for (int i = 0; i < __GRank__; i++) {
      int64_t* b_size = reinterpret_cast<int64_t*>(
          params[3 + i + param_permute[1] * size_struct]);
      std::cout << i << ":" << *b_size << std::endl;
    }
    std::cout << "D sizes." << std::endl;
    for (int i = 0; i < __GRank__; i++) {
      int64_t* d_size = reinterpret_cast<int64_t*>(
          params[3 + i + param_permute[2] * size_struct]);
      std::cout << i << ":" << *d_size << std::endl;
    }
  }

  std::vector<void*> ptr_batched_h(__GEMM_NUM__ * 3);
  void** ptr_batched_h_data = ptr_batched_h.data();
  void** ptr_A_batched_h = ptr_batched_h_data;
  void** ptr_B_batched_h = ptr_batched_h_data + __GEMM_NUM__;
  void** ptr_D_batched_h = ptr_batched_h_data + __GEMM_NUM__ * 2;
  for (int i = 0; i < __GEMM_NUM__; i++) {
    __ElementAType__* a = *reinterpret_cast<__ElementAType__**>(params[1 + param_permute[i * 3 + 0] * size_struct]);
    __ElementBType__* b = *reinterpret_cast<__ElementBType__**>(params[1 + param_permute[i * 3 + 1] * size_struct]);
    __ElementOutputType__* d =
        *reinterpret_cast<__ElementOutputType__**>(params[1 + param_permute[i * 3 + 2] * size_struct]);
    ptr_A_batched_h[i] = a;
    ptr_B_batched_h[i] = b;
    ptr_D_batched_h[i] = d;
  }
  void** ptr_batched_d;
  cudaMalloc((void**)&ptr_batched_d, sizeof(void*) * __GEMM_NUM__ * 3);
  cudaMemcpy(ptr_batched_d, ptr_batched_h_data, sizeof(void*) * __GEMM_NUM__ * 3, cudaMemcpyHostToDevice);

  int64_t m = std::is_same<__ElementALayout__, cutlass::layout::ColumnMajor>::value ?
      *reinterpret_cast<int64_t*>(params[3 + __GRank__ - 1]) :
      *reinterpret_cast<int64_t*>(params[3 + __GRank__ - 2]);
  int64_t k = std::is_same<__ElementALayout__, cutlass::layout::ColumnMajor>::value ?
      *reinterpret_cast<int64_t*>(params[3 + __GRank__ - 2]) :
      *reinterpret_cast<int64_t*>(params[3 + __GRank__ - 1]);
  int64_t n = std::is_same<__ElementBLayout__, cutlass::layout::ColumnMajor>::value ?
      *reinterpret_cast<int64_t*>(params[3 + size_struct + __GRank__ - 2]) :
      *reinterpret_cast<int64_t*>(params[3 + size_struct + __GRank__ - 1]);

  void** ptr_A_batched_d = ptr_batched_d;
  void** ptr_B_batched_d = ptr_batched_d + __GEMM_NUM__;
  void** ptr_D_batched_d = ptr_batched_d + __GEMM_NUM__ * 2;
  __SpecializedGemmFusion__ specialization(
      stream, batch_size, m, n, k,
      (const void**)ptr_A_batched_d, (const void**)ptr_B_batched_d,
      ptr_D_batched_d);
  return specialization.run();
};

)bladedisc_rstr"