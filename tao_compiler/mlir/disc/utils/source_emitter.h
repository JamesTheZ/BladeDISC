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
  std::string EmitElemWiseUnaryOp(Operation* op, ValueNameBinding& binding);
  std::string EmitElemWiseBinaryOp(Operation* op, ValueNameBinding& binding);
  std::string EmitElemWiseTernaryOp(Operation* op, ValueNameBinding& binding);

 private:
  llvm::DenseMap<std::string, int32_t> existing_names_;

 private:
  void EmitElemWiseOp(Operation* op, SmallVectorImpl<StringRef>& operands,
                      std::string& result, std::string& instruction);
  std::string EmitValueName(std::string op_str);
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // MLIR_DISC_UTILS_SOURCE_EMITTER