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
    %c0_i3 = hw.constant 0 : i3
    %false = hw.constant false
    %0 = comb.concat %mant_in, %false : i7, i1
    %u_lod8.y, %u_lod8.zero = hw.instance "u_lod8" @lod8(x: %0: i8) -> (y: i8, zero: i1) {sv.namehint = "data_one_hot"}
    hw.output %is_zero_in, %is_infinity_in, %is_nan_in, %is_inexact_in, %is_special_in, %c0_i3 : i1, i1, i1, i1, i1, i3
  }
}
