add_llvm_executable(equiv_miter equiv_miter.cpp)

target_link_libraries(equiv_miter
  PRIVATE
  EquivFusionToolEquivMiter
)

llvm_update_compile_flags(equiv_miter)
mlir_check_all_link_libraries(equiv_miter)


