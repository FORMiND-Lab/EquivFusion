set(LLVM_TARGET_DEFINITIONS Passes.td) 

mlir_tablegen(Passes.h.inc -gen-pass-decls -name EquivFusionRemoveRedundantFunc) 

mlir_tablegen(EquivFusionRemoveRedundantFunc.capi.h.inc -gen-pass-capi-header --prefix EquivFusionRemoveRedundantFunc)
mlir_tablegen(EquivFusionRemoveRedundantFunc.capi.cpp.inc -gen-pass-capi-impl --prefix EquivFusionRemoveRedundantFunc)

add_public_tablegen_target(EquivFusionRemoveRedundantFuncIncGen)
