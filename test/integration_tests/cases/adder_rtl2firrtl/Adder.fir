FIRRTL version 3.3.0
circuit Adder :%[[
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Adder|Adder",
    "group":"Adder"
  }
]]
  module Adder : @[src/main/scala/adder.scala 5:7]
    input a : UInt<8> @[src/main/scala/adder.scala 6:13]
    input b : UInt<8> @[src/main/scala/adder.scala 7:13]
    output sum : UInt<8> @[src/main/scala/adder.scala 8:15]

    node _sum_T = add(a, b) @[src/main/scala/adder.scala 9:12]
    node _sum_T_1 = tail(_sum_T, 1) @[src/main/scala/adder.scala 9:12]
    connect sum, _sum_T_1 @[src/main/scala/adder.scala 9:7]
