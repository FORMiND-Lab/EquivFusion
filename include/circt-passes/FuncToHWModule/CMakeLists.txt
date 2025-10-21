# 指定使用的td文件
set(LLVM_TARGET_DEFINITIONS Passes.td) 

# 生成命令基类实现，命令创建函数，命令注册函数，命令组注册函数的Passes.h.inc文件。-name指定命令组名称
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EquivFusionFuncToHWModule) 

# 生成供其他语言调用的capi定义文件
mlir_tablegen(EquivFusionFuncToHWModule.capi.h.inc -gen-pass-capi-header --prefix EquivFusionFuncToHWModule)
mlir_tablegen(EquivFusionFuncToHWModule.capi.cpp.inc -gen-pass-capi-impl --prefix EquivFusionFuncToHWModule)

add_public_tablegen_target(FuncToHWModuleIncGen)
