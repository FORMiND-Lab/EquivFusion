module {
  hw.module @lzc7_casex(in %mant_in : i7, in %is_zero_in : i1, in %is_infinity_in : i1, in %is_nan_in : i1, in %is_special_in : i1, in %is_inexact_in : i1, out is_zero_out : i1, out is_infinity_out : i1, out is_nan_out : i1, out is_inexact_out : i1, out is_special_out : i1, out lzc : i3) {
    %true = hw.constant true
    %c2_i7 = hw.constant 2 : i7
    %c-2_i7 = hw.constant -2 : i7
    %c4_i7 = hw.constant 4 : i7
    %c-4_i7 = hw.constant -4 : i7
    %c8_i7 = hw.constant 8 : i7
    %c-8_i7 = hw.constant -8 : i7
    %c16_i7 = hw.constant 16 : i7
    %c-16_i7 = hw.constant -16 : i7
    %c32_i7 = hw.constant 32 : i7
    %c-32_i7 = hw.constant -32 : i7
    %c-64_i7 = hw.constant -64 : i7
    %c-2_i3 = hw.constant -2 : i3
    %c-3_i3 = hw.constant -3 : i3
    %c-4_i3 = hw.constant -4 : i3
    %c3_i3 = hw.constant 3 : i3
    %c2_i3 = hw.constant 2 : i3
    %c1_i3 = hw.constant 1 : i3
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.and %mant_in, %c-64_i7 : i7
    %1 = comb.icmp eq %0, %c-64_i7 : i7
    %2 = comb.and %mant_in, %c-32_i7 : i7
    %3 = comb.icmp eq %2, %c32_i7 : i7
    %4 = comb.and %mant_in, %c-16_i7 : i7
    %5 = comb.icmp eq %4, %c16_i7 : i7
    %6 = comb.and %mant_in, %c-8_i7 : i7
    %7 = comb.icmp eq %6, %c8_i7 : i7
    %8 = comb.and %mant_in, %c-4_i7 : i7
    %9 = comb.icmp eq %8, %c4_i7 : i7
    %10 = comb.and %mant_in, %c-2_i7 : i7
    %11 = comb.icmp eq %10, %c2_i7 : i7
    %12 = comb.mux %11, %c-3_i3, %c-2_i3 : i3
    %13 = comb.xor %1, %true : i1
    %14 = comb.xor %3, %true : i1
    %15 = comb.and %14, %13 : i1
    %16 = comb.xor %5, %true : i1
    %17 = comb.and %16, %15 : i1
    %18 = comb.xor %7, %true : i1
    %19 = comb.and %18, %17, %9 : i1
    %20 = comb.mux %19, %c-4_i3, %12 : i3
    %21 = comb.and %17, %7 : i1
    %22 = comb.mux %21, %c3_i3, %20 : i3
    %23 = comb.and %15, %5 : i1
    %24 = comb.mux %23, %c2_i3, %22 : i3
    %25 = comb.and %13, %3 : i1
    %26 = comb.mux %25, %c1_i3, %24 : i3
    %27 = comb.or %is_special_in, %1 : i1
    %28 = comb.mux %27, %c0_i3, %26 : i3
    hw.output %is_zero_in, %is_infinity_in, %is_nan_in, %is_inexact_in, %is_special_in, %28 : i1, i1, i1, i1, i1, i3
  }
}
