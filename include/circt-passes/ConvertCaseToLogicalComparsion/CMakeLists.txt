set(LLVM_TARGET_DEFINITIONS Passes.td) 


mlir_tablegen(Passes.h.inc -gen-pass-decls -name EquivFusionConvertCaseToLogicalComparison) 

mlir_tablegen(EquivFusionConvertCaseToLogicalComparison.capi.h.inc -gen-pass-capi-header --prefix EquivFusionConvertCaseToLogicalComparison)
mlir_tablegen(EquivFusionConvertCaseToLogicalComparison.capi.cpp.inc -gen-pass-capi-impl --prefix EquivFusionConvertCaseToLogicalComparison)

add_public_tablegen_target(EquivFusionConvertCaseToLogicalComparisonIncGen)
