#ifndef EQUIVFUSION_FLATTEN_IO_ARRAY_PASSES_H
#define EQUIVFUSION_FLATTEN_IO_ARRAY_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {

#define GEN_PASS_DECL_EQUIVFUSIONFLATTENIOARRAY
#define GEN_PASS_REGISTRATION
#include "circt-passes/FlattenIOArray/Passes.h.inc"

}
#endif //EQUIVFUSION_FLATTEN_IO_ARRAY_PASSES_H
