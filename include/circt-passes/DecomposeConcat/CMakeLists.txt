set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EquivFusionDecomposeConcat)
mlir_tablegen(EquivFusionDecomposeConcat.capi.h.inc -gen-pass-capi-header --prefix EquivFusionDecomposeConcat)
mlir_tablegen(EquivFusionDecomposeConcat.capi.cpp.inc -gen-pass-capi-impl --prefix EquivFusionDecomposeConcat)
add_public_tablegen_target(EquivFusionDecomposeConcatIncGen)

