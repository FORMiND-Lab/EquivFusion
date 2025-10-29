module {
  func.func @LZC(%arg0: i8 {polygeist.param_name = "mant_in"}) -> i8 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8 = arith.constant 8 : index
    %c1_i8 = arith.constant 1 : i8
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %0 = arith.extsi %arg0 : i8 to i32
    %1:2 = affine.for %arg1 = 1 to 8 iter_args(%arg2 = %c0_i8, %arg3 = %c0_i8) -> (i8, i8) {
      %2 = arith.subi %c8, %arg1 : index
      %3 = arith.index_cast %2 : index to i32
      %4 = arith.cmpi eq, %arg2, %c0_i8 : i8
      %5 = arith.shrsi %0, %3 : i32
      %6 = arith.andi %5, %c1_i32 : i32
      %7 = arith.cmpi ne, %6, %c0_i32 : i32
      %8 = arith.select %7, %c1_i8, %arg2 : i8
      %9 = arith.select %4, %8, %arg2 : i8
      %10 = scf.if %4 -> (i8) {
        %11 = scf.if %7 -> (i8) {
          scf.yield %arg3 : i8
        } else {
          %12 = arith.addi %arg3, %c1_i8 : i8
          scf.yield %12 : i8
        }
        scf.yield %11 : i8
      } else {
        scf.yield %arg3 : i8
      }
      affine.yield %9, %10 : i8, i8
    }
    return %1#1 : i8
  }
}
