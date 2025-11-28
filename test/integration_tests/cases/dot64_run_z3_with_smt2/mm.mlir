module {
  func.func @mm(%arg0: memref<64xi16>, %arg1: memref<64xi16>) -> i48 {
    %c0_i64 = arith.constant 0 : i48
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
    affine.for %arg2 = 0 to 64 {
      %2 = affine.load %arg0[%arg2] : memref<64xi16>
      %3 = affine.load %arg1[%arg2] : memref<64xi16>
      %4 = arith.extsi %2 : i16 to i32
      %5 = arith.extsi %3 : i16 to i32
      %6 = arith.muli %4, %5 : i32
      affine.store %6, %alloc[%arg2] : memref<64xi32>
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<i48>
    affine.store %c0_i64, %alloc_0[] : memref<i48>
    affine.for %arg2 = 0 to 64 {
      %2 = affine.load %alloc[%arg2] : memref<64xi32>
      %3 = affine.load %alloc_0[] : memref<i48>
      %4 = arith.extsi %2 : i32 to i48
      %5 = arith.addi %4, %3 : i48
      affine.store %5, %alloc_0[] : memref<i48>
    }
    %0 = affine.load %alloc_0[] : memref<i48>
    return %0 : i48
  }
}

