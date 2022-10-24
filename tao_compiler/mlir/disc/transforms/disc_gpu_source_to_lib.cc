#include <fstream>
#if 1
#include <cstdlib>
#endif

#include "llvm/Support/Program.h"
// #include "llvm/Support/raw_ostream.h"
// #include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/util/env_var.h"

namespace mlir {
namespace disc_ral {

namespace {

template <typename T>
llvm::ArrayRef<T> AsArrayRef(const SmallVectorImpl<T>& vec) {
  return llvm::ArrayRef<T>(vec);
}

class DiscGPUSourceToLibPass
    : public DiscGPUSourceToLibPassBase<DiscGPUSourceToLibPass> {
 public:
  explicit DiscGPUSourceToLibPass(int cc_major, int cc_minor)
      : DiscGPUSourceToLibPassBase<
            DiscGPUSourceToLibPass>::DiscGPUSourceToLibPassBase() {
    cc_major_ = cc_major;
    cc_minor_ = cc_minor;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Find ops containing CUDA code and concatenate CUDA code.
    std::string cuda_code;
    SmallVector<lmhlo_disc::SourceCodeOp> source_code_ops;
    m.walk([&](lmhlo_disc::SourceCodeOp source_code_op) {
      source_code_ops.push_back(source_code_op);
    });

    for (auto source_code_op : source_code_ops) {
      if (auto attr = source_code_op.codeAttr()) {
        cuda_code += attr.getValue();
      } else {
        signalPassFailure();
        return;
      }
    }

    llvm::errs() << "[ZZ] cuda code to compile:\n" << cuda_code << "\n";

    // Compile CUDA code.
    std::string bin_path;
    compileCUDASource(cuda_code, bin_path);

    for (auto source_code_op : source_code_ops) {
      OpBuilder builder(source_code_op);
      source_code_op->setAttr(kDynLibPathAttr, builder.getStringAttr(bin_path));
    }
  }

 private:
  int cc_major_;
  int cc_minor_;

 private:
  LogicalResult executeProgram(const std::string& program,
                               const SmallVectorImpl<llvm::StringRef>& args,
                               const SmallVectorImpl<llvm::StringRef>& env);
  std::string getArchStr();
  LogicalResult compileCUDASource(const std::string& source,
                                  std::string& bin_path);
};

LogicalResult DiscGPUSourceToLibPass::executeProgram(
    const std::string& program, const SmallVectorImpl<llvm::StringRef>& args,
    const SmallVectorImpl<llvm::StringRef>& env) {
  std::string error_message;
  auto nvcc_args = AsArrayRef(args);
  std::string arg_str;
  for (auto arg : nvcc_args) {
    arg_str += arg.str() + " ";
  }
  std::string command = program + " " + arg_str;
#if 1
  llvm::errs() << "[ZZ] command: " << command << "\n";
#endif
  int result = std::system(command.c_str());
  if (result != 0) {
    return failure();
  } else {
    return success();
  }
  // TODO: use llvm ExecuteAndWait instead.
  // int result = llvm::sys::ExecuteAndWait(
  //     program, AsArrayRef(args), AsArrayRef(env), {}, 0, 0, &error_message);
  // return result ? failure() : success();
}

std::string DiscGPUSourceToLibPass::getArchStr() {
#if 1
  return "-gencode=arch=compute_86,code=sm_86";
#endif
  std::string arch = std::to_string(cc_major_) + std::to_string(cc_minor_);
  return "-gencode=arch=compute_" + arch + ",code=sm_" + arch;
}

LogicalResult DiscGPUSourceToLibPass::compileCUDASource(
    const std::string& source, std::string& bin_path) {
  std::string cuda_root;
  tensorflow::ReadStringFromEnvVar("DISC_CUDA_ROOT", "/usr/local/cuda/",
                                   &cuda_root);
  std::string nvcc_path = cuda_root + "bin";
  auto nvcc_program = llvm::sys::findProgramByName("nvcc", {nvcc_path});
  if (!nvcc_program) {
    return failure();
  }
  VLOG(2) << "nvcc found in path: " << *nvcc_program;

  std::string random_number = std::to_string(tensorflow::random::New64());
  std::string tmp_path = "/tmp/";
  std::string source_path = tmp_path + random_number + ".cu";
  bin_path = tmp_path + random_number + ".so";

  std::ofstream cufile;
  cufile.open(source_path);
  cufile << source;
  cufile.close();

  std::string disc_root;
  tensorflow::ReadStringFromEnvVar(
      "DISC_ROOT", "/home/zhengzhen/workspace/dev/BladeDISC", &disc_root);
  if (disc_root.empty()) {
    return failure();
  }
  std::string cutlass_root = disc_root + "/tao/third_party/cutlass";
  std::string cutlass_inc = "-I" + cutlass_root + "/include";
  std::string util_inc = "-I" + cutlass_root + "/tools/util/include";
  std::string lib_inc = "-I" + cutlass_root + "/tools/library/include";
  std::string cuda_lib = "-L" + cuda_root + "/lib64";

  std::string opt_level = "-O3";
  std::string arch_str = getArchStr();
  SmallVector<llvm::StringRef> nvcc_args{llvm::StringRef(opt_level),
                                         llvm::StringRef("--std=c++11"),
                                         llvm::StringRef(cutlass_inc),
                                         llvm::StringRef(util_inc),
                                         llvm::StringRef(lib_inc),
                                         llvm::StringRef(cuda_lib),
                                         llvm::StringRef(arch_str),
                                         llvm::StringRef("-shared"),
                                         llvm::StringRef("--compiler-options"),
                                         llvm::StringRef("'-fPIC'"),
                                         llvm::StringRef("-o"),
                                         llvm::StringRef(bin_path),
                                         llvm::StringRef(source_path)};

  SmallVector<llvm::StringRef> nvcc_env{
      llvm::StringRef(
          "LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64"),
      llvm::StringRef("PATH=/usr/local/"
                      "sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
  };

  if (failed(executeProgram(*nvcc_program, nvcc_args, nvcc_env))) {
    return failure();
  }
  return success();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscGPUSourceToLibPass(
    int cc_major, int cc_minor) {
  return std::make_unique<DiscGPUSourceToLibPass>(cc_major, cc_minor);
}

}  // namespace disc_ral
}  // namespace mlir