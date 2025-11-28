# 指定使用的td文件
set(LLVM_TARGET_DEFINITIONS Passes.td) 

# 生成命令基类实现，命令创建函数，命令注册函数，命令组注册函数的Passes.h.inc文件。-name指定命令组名称
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EquivFusionMemrefHLS) 

# 生成供其他语言调用的capi定义文件
mlir_tablegen(EquivFusionMemrefHLS.capi.h.inc -gen-pass-capi-header --prefix EquivFusionMemrefHLS)
mlir_tablegen(EquivFusionMemrefHLS.capi.cpp.inc -gen-pass-capi-impl --prefix EquivFusionMemrefHLS)

add_public_tablegen_target(EquivFusionMemrefHLSIncGen)
