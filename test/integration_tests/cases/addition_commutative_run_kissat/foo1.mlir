hw.module @foo1(in %a : i8, in %b : i8, out c : i8) {
  %add = comb.add %a, %b: i8
  hw.output %add : i8
}