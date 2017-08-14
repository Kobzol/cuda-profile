#include "Values.h"
#include "Types.h"

#include <llvm/IR/Constants.h>

using namespace llvm;

Value* Values::int32(Module* module, int32_t value)
{
    return ConstantInt::get(Types::int32(module), value, false);
}
Value* Values::int64(Module* module, int64_t value)
{
    return ConstantInt::get(Types::int64(module), value, false);
}
