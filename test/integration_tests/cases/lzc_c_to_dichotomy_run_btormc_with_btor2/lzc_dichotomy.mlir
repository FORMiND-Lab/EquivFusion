module {
  hw.module @lzc7_dichotomy(in %in_0 : i8, out out_0 : i8) {
    %c-1_i3 = hw.constant -1 : i3
    %c0_i2 = hw.constant 0 : i2
    %c0_i4 = hw.constant 0 : i4
    %c0_i5 = hw.constant 0 : i5
    %0 = comb.concat %2, %7, %14 : i1, i1, i1
    %1 = comb.extract %in_0 from 4 : (i8) -> i4
    %2 = comb.icmp ne %1, %c0_i4 : i4
    %3 = comb.extract %in_0 from 6 : (i8) -> i2
    %4 = comb.icmp ne %3, %c0_i2 : i2
    %5 = comb.extract %in_0 from 2 : (i8) -> i2
    %6 = comb.icmp ne %5, %c0_i2 : i2
    %7 = comb.mux %2, %4, %6 : i1
    %8 = comb.extract %in_0 from 7 : (i8) -> i1
    %9 = comb.extract %in_0 from 5 : (i8) -> i1
    %10 = comb.mux %4, %8, %9 : i1
    %11 = comb.extract %in_0 from 3 : (i8) -> i1
    %12 = comb.extract %in_0 from 1 : (i8) -> i1
    %13 = comb.mux %6, %11, %12 : i1
    %14 = comb.mux %2, %10, %13 : i1
    %15 = comb.xor %0, %c-1_i3 : i3
    %16 = comb.concat %c0_i5, %15 : i5, i3
    hw.output %16 : i8
  }
}
