hw.module @foo2(in %a : i8, in %b : i8, out c : i8) {
  %add = comb.add %b, %a: i8
  hw.output %add : i8
}