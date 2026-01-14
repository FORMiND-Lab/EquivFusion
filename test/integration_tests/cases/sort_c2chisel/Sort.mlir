module {
  func.func @Sort(%arg0: memref<8xi8> {polygeist.param_name = "input"}, %arg1: memref<8xi8> {polygeist.param_name = "output"}) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8 = arith.constant 8 : index
    %alloca = memref.alloca() : memref<8xi8>
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] : memref<8xi8>
      affine.store %0, %alloca[%arg2] : memref<8xi8>
    }
    affine.for %arg2 = 1 to 8 {
      %0 = arith.subi %c8, %arg2 : index
      %1 = arith.index_cast %0 : index to i32
      %2 = affine.load %alloca[0] : memref<8xi8>
      %3 = affine.for %arg3 = 1 to 8 iter_args(%arg4 = %2) -> (i8) {
        %4 = arith.index_cast %arg3 : index to i32
        %5 = arith.cmpi ule, %4, %1 : i32
        %6:2 = scf.if %5 -> (i8, i8) {
          %7 = affine.load %alloca[%arg3] : memref<8xi8>
          %8 = arith.extui %7 : i8 to i32
          %9 = arith.extui %arg4 : i8 to i32
          %10 = arith.cmpi sgt, %8, %9 : i32
          %11 = arith.select %10, %arg4, %7 : i8
          %12 = arith.select %10, %7, %arg4 : i8
          scf.yield %11, %12 : i8, i8
        } else {
          %7 = affine.load %alloca[%arg3 - 1] : memref<8xi8>
          scf.yield %7, %arg4 : i8, i8
        }
        affine.store %6#0, %alloca[%arg3 - 1] : memref<8xi8>
        affine.yield %6#1 : i8
      }
      affine.store %3, %alloca[-%arg2 + 8] : memref<8xi8>
    }
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %alloca[%arg2] : memref<8xi8>
      affine.store %0, %arg1[%arg2] : memref<8xi8>
    }
    return
  }
}
