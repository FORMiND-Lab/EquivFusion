FIRRTL version 3.3.0
circuit Sort :%[[
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_1",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_2",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_3",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_4",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_5",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_6",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_7",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_8",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_9",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_10",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_11",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_12",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_13",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_14",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_15",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_16",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_17",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_18",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_19",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_20",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_21",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_22",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|CompareAndSwap_23",
    "group":"CompareAndSwap"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~Sort|Sort",
    "group":"Sort"
  }
]]
  module CompareAndSwap : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_1 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_2 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_3 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_4 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_5 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_6 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_7 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_8 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_9 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_10 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_11 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_12 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_13 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_14 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_15 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_16 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_17 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_18 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_19 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_20 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_21 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_22 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module CompareAndSwap_23 : @[src/main/scala/sort.scala 6:7]
    output io : { flip a : UInt<8>, flip b : UInt<8>, min : UInt<8>, max : UInt<8>} @[src/main/scala/sort.scala 7:14]

    node _T = leq(io.a, io.b) @[src/main/scala/sort.scala 14:13]
    when _T : @[src/main/scala/sort.scala 14:22]
      connect io.min, io.a @[src/main/scala/sort.scala 15:12]
      connect io.max, io.b @[src/main/scala/sort.scala 16:12]
    else :
      connect io.min, io.b @[src/main/scala/sort.scala 18:12]
      connect io.max, io.a @[src/main/scala/sort.scala 19:12]


  module Sort : @[src/main/scala/sort.scala 23:7]
    input input : UInt<8>[8] @[src/main/scala/sort.scala 24:17]
    output output : UInt<8>[8] @[src/main/scala/sort.scala 25:18]

    inst m of CompareAndSwap @[src/main/scala/sort.scala 28:19]
    connect m.io.a, input[0] @[src/main/scala/sort.scala 29:12]
    connect m.io.b, input[1] @[src/main/scala/sort.scala 30:12]
    inst m_1 of CompareAndSwap_1 @[src/main/scala/sort.scala 28:19]
    connect m_1.io.a, input[2] @[src/main/scala/sort.scala 29:12]
    connect m_1.io.b, input[3] @[src/main/scala/sort.scala 30:12]
    inst m_2 of CompareAndSwap_2 @[src/main/scala/sort.scala 28:19]
    connect m_2.io.a, input[4] @[src/main/scala/sort.scala 29:12]
    connect m_2.io.b, input[5] @[src/main/scala/sort.scala 30:12]
    inst m_3 of CompareAndSwap_3 @[src/main/scala/sort.scala 28:19]
    connect m_3.io.a, input[6] @[src/main/scala/sort.scala 29:12]
    connect m_3.io.b, input[7] @[src/main/scala/sort.scala 30:12]
    inst m_4 of CompareAndSwap_4 @[src/main/scala/sort.scala 28:19]
    connect m_4.io.a, m.io.min @[src/main/scala/sort.scala 29:12]
    connect m_4.io.b, m_1.io.max @[src/main/scala/sort.scala 30:12]
    inst m_5 of CompareAndSwap_5 @[src/main/scala/sort.scala 28:19]
    connect m_5.io.a, m.io.max @[src/main/scala/sort.scala 29:12]
    connect m_5.io.b, m_1.io.min @[src/main/scala/sort.scala 30:12]
    inst m_6 of CompareAndSwap_6 @[src/main/scala/sort.scala 28:19]
    connect m_6.io.a, m_4.io.min @[src/main/scala/sort.scala 29:12]
    connect m_6.io.b, m_5.io.min @[src/main/scala/sort.scala 30:12]
    inst m_7 of CompareAndSwap_7 @[src/main/scala/sort.scala 28:19]
    connect m_7.io.a, m_4.io.max @[src/main/scala/sort.scala 29:12]
    connect m_7.io.b, m_5.io.max @[src/main/scala/sort.scala 30:12]
    inst m_8 of CompareAndSwap_8 @[src/main/scala/sort.scala 28:19]
    connect m_8.io.a, m_2.io.min @[src/main/scala/sort.scala 29:12]
    connect m_8.io.b, m_3.io.max @[src/main/scala/sort.scala 30:12]
    inst m_9 of CompareAndSwap_9 @[src/main/scala/sort.scala 28:19]
    connect m_9.io.a, m_2.io.max @[src/main/scala/sort.scala 29:12]
    connect m_9.io.b, m_3.io.min @[src/main/scala/sort.scala 30:12]
    inst m_10 of CompareAndSwap_10 @[src/main/scala/sort.scala 28:19]
    connect m_10.io.a, m_8.io.max @[src/main/scala/sort.scala 29:12]
    connect m_10.io.b, m_9.io.max @[src/main/scala/sort.scala 30:12]
    inst m_11 of CompareAndSwap_11 @[src/main/scala/sort.scala 28:19]
    connect m_11.io.a, m_8.io.min @[src/main/scala/sort.scala 29:12]
    connect m_11.io.b, m_9.io.min @[src/main/scala/sort.scala 30:12]
    inst m_12 of CompareAndSwap_12 @[src/main/scala/sort.scala 28:19]
    connect m_12.io.a, m_6.io.min @[src/main/scala/sort.scala 29:12]
    connect m_12.io.b, m_10.io.max @[src/main/scala/sort.scala 30:12]
    inst m_13 of CompareAndSwap_13 @[src/main/scala/sort.scala 28:19]
    connect m_13.io.a, m_6.io.max @[src/main/scala/sort.scala 29:12]
    connect m_13.io.b, m_10.io.min @[src/main/scala/sort.scala 30:12]
    inst m_14 of CompareAndSwap_14 @[src/main/scala/sort.scala 28:19]
    connect m_14.io.a, m_7.io.min @[src/main/scala/sort.scala 29:12]
    connect m_14.io.b, m_11.io.max @[src/main/scala/sort.scala 30:12]
    inst m_15 of CompareAndSwap_15 @[src/main/scala/sort.scala 28:19]
    connect m_15.io.a, m_7.io.max @[src/main/scala/sort.scala 29:12]
    connect m_15.io.b, m_11.io.min @[src/main/scala/sort.scala 30:12]
    inst m_16 of CompareAndSwap_16 @[src/main/scala/sort.scala 28:19]
    connect m_16.io.a, m_12.io.min @[src/main/scala/sort.scala 29:12]
    connect m_16.io.b, m_14.io.min @[src/main/scala/sort.scala 30:12]
    inst m_17 of CompareAndSwap_17 @[src/main/scala/sort.scala 28:19]
    connect m_17.io.a, m_13.io.min @[src/main/scala/sort.scala 29:12]
    connect m_17.io.b, m_15.io.min @[src/main/scala/sort.scala 30:12]
    inst m_18 of CompareAndSwap_18 @[src/main/scala/sort.scala 28:19]
    connect m_18.io.a, m_12.io.max @[src/main/scala/sort.scala 29:12]
    connect m_18.io.b, m_14.io.max @[src/main/scala/sort.scala 30:12]
    inst m_19 of CompareAndSwap_19 @[src/main/scala/sort.scala 28:19]
    connect m_19.io.a, m_13.io.max @[src/main/scala/sort.scala 29:12]
    connect m_19.io.b, m_15.io.max @[src/main/scala/sort.scala 30:12]
    inst m_20 of CompareAndSwap_20 @[src/main/scala/sort.scala 28:19]
    connect m_20.io.a, m_16.io.min @[src/main/scala/sort.scala 29:12]
    connect m_20.io.b, m_17.io.min @[src/main/scala/sort.scala 30:12]
    inst m_21 of CompareAndSwap_21 @[src/main/scala/sort.scala 28:19]
    connect m_21.io.a, m_16.io.max @[src/main/scala/sort.scala 29:12]
    connect m_21.io.b, m_17.io.max @[src/main/scala/sort.scala 30:12]
    inst m_22 of CompareAndSwap_22 @[src/main/scala/sort.scala 28:19]
    connect m_22.io.a, m_18.io.min @[src/main/scala/sort.scala 29:12]
    connect m_22.io.b, m_19.io.min @[src/main/scala/sort.scala 30:12]
    inst m_23 of CompareAndSwap_23 @[src/main/scala/sort.scala 28:19]
    connect m_23.io.a, m_18.io.max @[src/main/scala/sort.scala 29:12]
    connect m_23.io.b, m_19.io.max @[src/main/scala/sort.scala 30:12]
    connect output[0], m_23.io.max @[src/main/scala/sort.scala 67:13]
    connect output[1], m_23.io.min @[src/main/scala/sort.scala 68:13]
    connect output[2], m_22.io.max @[src/main/scala/sort.scala 69:13]
    connect output[3], m_22.io.min @[src/main/scala/sort.scala 70:13]
    connect output[4], m_21.io.max @[src/main/scala/sort.scala 71:13]
    connect output[5], m_21.io.min @[src/main/scala/sort.scala 72:13]
    connect output[6], m_20.io.max @[src/main/scala/sort.scala 73:13]
    connect output[7], m_20.io.min @[src/main/scala/sort.scala 74:13]

