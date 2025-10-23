module {
  func.func @increment(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.addi %arg0, %c1_i32 : i32
    return %0 : i32
  }
  func.func @test(%arg0: i8, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg0, %c0_i8 : i8
    %1 = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %c0_i32) -> (i32) {
      %2 = scf.if %0 -> (i32) {
        %3 = arith.addi %arg3, %arg1 : i32
        scf.yield %3 : i32
      } else {
        %3 = arith.addi %arg3, %c1_i32 : i32
        scf.yield %3 : i32
      }
      affine.yield %2 : i32
    }
    return %1 : i32
  }
}
