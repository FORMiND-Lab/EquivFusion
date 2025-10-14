module {
  hw.module @lzc7_dichotomy(in %sign_in : i1, in %exp_in : i6, in %mant_in : i8, in %is_zero_in : i1, in %is_infinity_in : i1, in %is_nan_in : i1, in %is_special_in : i1, in %is_inexact_in : i1, in %is_zero_a_in : i1, in %is_zero_b_in : i1, out is_zero_a_out : i1, out is_zero_b_out : i1, out is_zero_out : i1, out is_infinity_out : i1, out is_nan_out : i1, out is_inexact_out : i1, out is_special_out : i1, out mant_out : i8, out sign_out : i1, out exp_out : i6, out lzc : i3) {
    %c-1_i3 = hw.constant -1 : i3
    %c0_i2 = hw.constant 0 : i2
    %c0_i4 = hw.constant 0 : i4
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %2, %7, %14 : i1, i1, i1
    %1 = comb.extract %mant_in from 4 : (i8) -> i4
    %2 = comb.icmp ne %1, %c0_i4 : i4
    %3 = comb.extract %mant_in from 6 : (i8) -> i2
    %4 = comb.icmp ne %3, %c0_i2 : i2
    %5 = comb.extract %mant_in from 2 : (i8) -> i2
    %6 = comb.icmp ne %5, %c0_i2 : i2
    %7 = comb.mux %2, %4, %6 : i1
    %8 = comb.extract %mant_in from 7 : (i8) -> i1
    %9 = comb.extract %mant_in from 5 : (i8) -> i1
    %10 = comb.mux %4, %8, %9 : i1
    %11 = comb.extract %mant_in from 3 : (i8) -> i1
    %12 = comb.extract %mant_in from 1 : (i8) -> i1
    %13 = comb.mux %6, %11, %12 : i1
    %14 = comb.mux %2, %10, %13 : i1
    %15 = comb.xor %0, %c-1_i3 : i3
    %16 = comb.mux %is_special_in, %c0_i3, %15 : i3
    hw.output %is_zero_a_in, %is_zero_b_in, %is_zero_in, %is_infinity_in, %is_nan_in, %is_inexact_in, %is_special_in, %mant_in, %sign_in, %exp_in, %16 : i1, i1, i1, i1, i1, i1, i1, i8, i1, i6, i3
  }
}
