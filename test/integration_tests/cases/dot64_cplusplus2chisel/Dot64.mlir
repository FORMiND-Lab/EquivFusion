module {
  func.func @Dot64(%arg0: memref<64xi16> {polygeist.param_name = "arg_0"}, %arg1: memref<64xi16> {polygeist.param_name = "arg_1"}) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i64 = arith.constant 0 : i64
    %0 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %c0_i64) -> (i64) {
      %1 = affine.load %arg0[%arg2] : memref<64xi16>
      %2 = arith.extsi %1 : i16 to i32
      %3 = affine.load %arg1[%arg2] : memref<64xi16>
      %4 = arith.extsi %3 : i16 to i32
      %5 = arith.muli %2, %4 : i32
      %6 = arith.extsi %5 : i32 to i64
      %7 = arith.addi %arg3, %6 : i64
      affine.yield %7 : i64
    }
    return %0 : i64
  }
}
