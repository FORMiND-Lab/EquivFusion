module {
  hw.module private @lod4(in %x : i4, out y : i4, out zero : i1) {
    %true = hw.constant true
    %c0_i4 = hw.constant 0 : i4
    %0 = comb.concat %1, %4, %8, %11 : i1, i1, i1, i1
    %1 = comb.extract %x from 3 : (i4) -> i1
    %2 = comb.xor %1, %true : i1
    %3 = comb.extract %x from 2 : (i4) -> i1
    %4 = comb.and %2, %3 : i1
    %5 = comb.xor %3, %true : i1
    %6 = comb.and %2, %5 : i1
    %7 = comb.extract %x from 1 : (i4) -> i1
    %8 = comb.and %6, %7 : i1
    %9 = comb.xor %7, %true : i1
    %10 = comb.extract %x from 0 : (i4) -> i1
    %11 = comb.and %6, %9, %10 : i1
    %12 = comb.icmp eq %x, %c0_i4 : i4
    hw.output %0, %12 : i4, i1
  }
  hw.module private @lod8(in %x : i8, out y : i8, out zero : i1) {
    %c0_i4 = hw.constant 0 : i4
    %0 = comb.extract %x from 4 : (i8) -> i4
    %u_hi.y, %u_hi.zero = hw.instance "u_hi" @lod4(x: %0: i4) -> (y: i4, zero: i1) {sv.namehint = "z_hi"}
    %1 = comb.extract %x from 0 : (i8) -> i4
    %u_lo.y, %u_lo.zero = hw.instance "u_lo" @lod4(x: %1: i4) -> (y: i4, zero: i1) {sv.namehint = "z_lo"}
    %2 = comb.concat %c0_i4, %u_lo.y : i4, i4
    %3 = comb.concat %u_hi.y, %c0_i4 : i4, i4
    %4 = comb.mux %u_hi.zero, %2, %3 : i8
    %5 = comb.and %u_hi.zero, %u_lo.zero : i1
    hw.output %4, %5 : i8, i1
  }
  hw.module @lzc7_loop(in %mant_in : i7, in %is_zero_in : i1, in %is_infinity_in : i1, in %is_nan_in : i1, in %is_special_in : i1, in %is_inexact_in : i1, out is_zero_out : i1, out is_infinity_out : i1, out is_nan_out : i1, out is_inexact_out : i1, out is_special_out : i1, out lzc : i3) {
    %c0_i2 = hw.constant 0 : i2
    %c1_i7 = hw.constant 1 : i7
    %c1_i3 = hw.constant 1 : i3
    %c-1_i7 = hw.constant -1 : i7
    %c0_i3 = hw.constant 0 : i3
    %false = hw.constant false
    %0 = comb.concat %mant_in, %false : i7, i1
    %u_lod8.y, %u_lod8.zero = hw.instance "u_lod8" @lod8(x: %0: i8) -> (y: i8, zero: i1) {sv.namehint = "data_one_hot"}
    %1 = comb.extract %u_lod8.y from 1 : (i8) -> i7
    %2 = comb.xor %1, %c-1_i7 : i7
    %3 = comb.add %2, %c1_i7 : i7
    %4 = comb.xor %3, %1 {sv.namehint = "data_zero_two_one"} : i7
    %5 = comb.extract %4 from 6 : (i7) -> i1
    %6 = comb.concat %c0_i2, %5 : i2, i1
    %7 = comb.extract %4 from 5 : (i7) -> i1
    %8 = comb.add %6, %c1_i3 : i3
    %9 = comb.mux %7, %8, %6 : i3
    %10 = comb.extract %4 from 4 : (i7) -> i1
    %11 = comb.add %9, %c1_i3 : i3
    %12 = comb.mux %10, %11, %9 : i3
    %13 = comb.extract %4 from 3 : (i7) -> i1
    %14 = comb.add %12, %c1_i3 : i3
    %15 = comb.mux %13, %14, %12 : i3
    %16 = comb.extract %4 from 2 : (i7) -> i1
    %17 = comb.add %15, %c1_i3 : i3
    %18 = comb.mux %16, %17, %15 : i3
    %19 = comb.extract %4 from 1 : (i7) -> i1
    %20 = comb.add %18, %c1_i3 : i3
    %21 = comb.mux %19, %20, %18 : i3
    %22 = comb.extract %4 from 0 : (i7) -> i1
    %23 = comb.add %21, %c1_i3 : i3
    %24 = comb.mux %22, %23, %21 : i3
    %25 = comb.mux %is_special_in, %c0_i3, %24 : i3
    hw.output %is_zero_in, %is_infinity_in, %is_nan_in, %is_inexact_in, %is_special_in, %25 : i1, i1, i1, i1, i1, i3
  }
}
