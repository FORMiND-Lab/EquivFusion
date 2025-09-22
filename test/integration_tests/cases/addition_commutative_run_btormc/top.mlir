module {
  hw.module @top(in %a : i8, in %b : i8) {
    %0 = comb.add %a, %b : i8
    %1 = comb.add %b, %a : i8
      %2 = comb.icmp eq %0, %1 : i8
      verif.assert %2 label "" : i1
    hw.output
  }
}
