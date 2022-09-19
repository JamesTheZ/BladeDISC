#ifndef MLIR_DISC_UTILS_SOURCE_EMITTER
#define MLIR_DISC_UTILS_SOURCE_EMITTER

#include <string>

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace disc_ral {

class SourceEmitterCUDA {
 public:
  using ValueNameBinding = llvm::DenseMap<Value, std::string>;

 public:
  llvm::Optional<std::string> EmitElemWiseUnaryOp(Operation* op,
                                                  ValueNameBinding binding);

  llvm::Optional<std::string> EmitElemWiseBinaryOp(Operation* op,
                                                   ValueNameBinding binding);

  llvm::Optional<std::string> EmitElemWiseTernaryOp(Operation* op,
                                                    ValueNameBinding& binding);

  void initValueNameBinding(SmallVectorImpl<Value> inputs,
                            SmallVectorImpl<std::string> names,
                            ValueNameBinding& binding);

 private:
  std::unordered_map<std::string, int32_t> existing_names_;

 private:
  std::string EmitUniqueName(llvm::StringRef op_str);
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // MLIR_DISC_UTILS_SOURCE_EMITTER