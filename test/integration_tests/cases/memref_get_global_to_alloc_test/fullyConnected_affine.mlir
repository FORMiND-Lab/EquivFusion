module {
  memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<0.241402462> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x3xf32 : memref<1x3xf32> = dense_resource<torch_tensor_1_3_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xf32_1 : memref<3xf32> = dense_resource<torch_tensor_3_torch.float32_1> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3x3xf32_0 : memref<3x3xf32> = dense_resource<torch_tensor_3_3_torch.float32_1> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3x3xf32 : memref<3x3xf32> = dense_resource<torch_tensor_3_3_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xf32 : memref<3xf32> = dense_resource<torch_tensor_3_torch.float32> {alignment = 64 : i64}
  func.func @fullyConnected(%arg0: memref<1x3xf32>) -> memref<1x1xf32> {
    %cst = arith.constant 0.241402462 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_3xf32 : memref<3xf32>
    %1 = memref.get_global @__constant_3x3xf32 : memref<3x3xf32>
    %2 = memref.get_global @__constant_3x3xf32_0 : memref<3x3xf32>
    %3 = memref.get_global @__constant_3xf32_1 : memref<3xf32>
    %4 = memref.get_global @__constant_1x3xf32 : memref<1x3xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    affine.for %arg1 = 0 to 3 {
      affine.for %arg2 = 0 to 3 {
        %5 = affine.load %1[%arg2, %arg1] : memref<3x3xf32>
        affine.store %5, %alloc[%arg1, %arg2] : memref<3x3xf32>
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x3xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x3xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        affine.store %cst_0, %alloc_2[%arg1, %arg2] : memref<1x3xf32>
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x3xf32>
    memref.copy %alloc_2, %alloc_3 : memref<1x3xf32> to memref<1x3xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 3 {
          %5 = affine.load %arg0[%arg1, %arg3] : memref<1x3xf32>
          %6 = affine.load %alloc[%arg3, %arg2] : memref<3x3xf32>
          %7 = affine.load %alloc_3[%arg1, %arg2] : memref<1x3xf32>
          %8 = arith.mulf %5, %6 : f32
          %9 = arith.addf %7, %8 : f32
          affine.store %9, %alloc_3[%arg1, %arg2] : memref<1x3xf32>
        }
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        %5 = affine.load %alloc_3[%arg1, %arg2] : memref<1x3xf32>
        %6 = affine.load %0[%arg2] : memref<3xf32>
        %7 = arith.addf %5, %6 : f32
        affine.store %7, %alloc_1[%arg1, %arg2] : memref<1x3xf32>
      }
    }
    affine.for %arg1 = 0 to 3 {
      affine.for %arg2 = 0 to 3 {
        %5 = affine.load %2[%arg2, %arg1] : memref<3x3xf32>
        affine.store %5, %alloc[%arg1, %arg2] : memref<3x3xf32>
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 3 {
          %5 = affine.load %alloc_1[%arg1, %arg3] : memref<1x3xf32>
          %6 = affine.load %alloc[%arg3, %arg2] : memref<3x3xf32>
          %7 = affine.load %alloc_2[%arg1, %arg2] : memref<1x3xf32>
          %8 = arith.mulf %5, %6 : f32
          %9 = arith.addf %7, %8 : f32
          affine.store %9, %alloc_2[%arg1, %arg2] : memref<1x3xf32>
        }
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        %5 = affine.load %alloc_2[%arg1, %arg2] : memref<1x3xf32>
        %6 = affine.load %3[%arg2] : memref<3xf32>
        %7 = arith.addf %5, %6 : f32
        affine.store %7, %alloc_1[%arg1, %arg2] : memref<1x3xf32>
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<3x1xf32>
    affine.for %arg1 = 0 to 3 {
      affine.for %arg2 = 0 to 1 {
        %5 = affine.load %4[%arg2, %arg1] : memref<1x3xf32>
        affine.store %5, %alloc_4[%arg1, %arg2] : memref<3x1xf32>
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.store %cst_0, %alloc_5[%arg1, %arg2] : memref<1x1xf32>
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 3 {
          %5 = affine.load %alloc_1[%arg1, %arg3] : memref<1x3xf32>
          %6 = affine.load %alloc_4[%arg3, %arg2] : memref<3x1xf32>
          %7 = affine.load %alloc_5[%arg1, %arg2] : memref<1x1xf32>
          %8 = arith.mulf %5, %6 : f32
          %9 = arith.addf %7, %8 : f32
          affine.store %9, %alloc_5[%arg1, %arg2] : memref<1x1xf32>
        }
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        %5 = affine.load %alloc_5[%arg1, %arg2] : memref<1x1xf32>
        %6 = arith.addf %5, %cst : f32
        affine.store %6, %alloc_5[%arg1, %arg2] : memref<1x1xf32>
      }
    }
    return %alloc_5 : memref<1x1xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_3_torch.float32: "0x04000000CDCEE5BE6599033FBD3523BE",
      torch_tensor_3_torch.float32_1: "0x04000000101D123E342E5DBE930BD7BD",
      torch_tensor_3_3_torch.float32_1: "0x0400000076DC093FD304EC3D47B7A03CAED8FFBE9324103F9BB9873EEE50823EB9864E3EBDFB793E",
      torch_tensor_3_3_torch.float32: "0x04000000C94E753E24B505BF9813CB3E822F023ED090523E1EB12FBE32CCB53ED3F411BF4850B73E",
      torch_tensor_3_torch.float32: "0x040000000ADD02BD3238A7BE3AD4463E"
    }
  }
#-}

