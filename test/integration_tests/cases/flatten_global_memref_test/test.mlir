module {
  memref.global constant @a : memref<2x2xi32> = dense<[[1, 2], [3, 4]]>
  func.func @test() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = memref.get_global @a : memref<2x2xi32>
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %1 = memref.load %0[%c1, %c1_0] : memref<2x2xi32>
    return %1 : i32
  }
}

