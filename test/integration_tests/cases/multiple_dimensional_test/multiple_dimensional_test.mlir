module {
  func.func @test(%arg0: memref<2x2xi32> {polygeist.param_name = "data"}) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg1 = 0 to 2 {
      affine.for %arg2 = 0 to 2 {
        %1 = affine.load %alloca[] : memref<i32>
        %2 = affine.load %arg0[%arg1, %arg2] : memref<2x2xi32>
        %3 = arith.addi %1, %2 : i32
        affine.store %3, %alloca[] : memref<i32>
      }
    }
    %0 = affine.load %alloca[] : memref<i32>
    return %0 : i32
  }
}
