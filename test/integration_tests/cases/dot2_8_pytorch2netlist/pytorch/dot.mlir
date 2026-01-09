module {
  func.func @dot(%arg0: memref<2xi8>, %arg1: memref<2xi8>) -> memref<i8> {
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xi8>
    affine.for %arg2 = 0 to 2 {
      %2 = affine.load %arg0[%arg2] : memref<2xi8>
      %3 = affine.load %arg1[%arg2] : memref<2xi8>
      %4 = arith.muli %2, %3 : i8
      affine.store %4, %alloc[%arg2] : memref<2xi8>
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<i64>
    affine.store %c0_i64, %alloc_0[] : memref<i64>
    affine.for %arg2 = 0 to 2 {
      %2 = affine.load %alloc[%arg2] : memref<2xi8>
      %3 = affine.load %alloc_0[] : memref<i64>
      %4 = arith.extsi %2 : i8 to i64
      %5 = arith.addi %4, %3 : i64
      affine.store %5, %alloc_0[] : memref<i64>
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<i8>
    %0 = affine.load %alloc_0[] : memref<i64>
    %1 = arith.trunci %0 : i64 to i8
    affine.store %1, %alloc_1[] : memref<i8>
    return %alloc_1 : memref<i8>
  }
}

