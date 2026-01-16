import chisel3._
// IMPORTANT: don't `import chisel3.util._` here, because it introduces
// `chisel3.util.circt` and would shadow the top-level `circt` package.
import circt.stage.ChiselStage

class CompareAndSwap(width: Int) extends RawModule {
  val io = IO(new Bundle {
    val a = Input(UInt(width.W))
    val b = Input(UInt(width.W))
    val min = Output(UInt(width.W))
    val max = Output(UInt(width.W))
  })

  when(io.a <= io.b) {
    io.min := io.a
    io.max := io.b
  }.otherwise {
    io.min := io.b
    io.max := io.a
  }
}

class Sort(width: Int = 8) extends RawModule {
  val input = IO(Input(Vec(8, UInt(width.W))))
  val output = IO(Output(Vec(8, UInt(width.W))))

  def CAS(a: UInt, b: UInt): (UInt, UInt) = {
    val m = Module(new CompareAndSwap(width))
    m.io.a := a
    m.io.b := b
    (m.io.min, m.io.max)
  }

  // === Stage 1 ===
  val (s1_0, s1_1) = CAS(input(0), input(1))
  val (s1_3, s1_2) = CAS(input(2), input(3)) // Swap order for bitonic merge
  val (s1_4, s1_5) = CAS(input(4), input(5))
  val (s1_7, s1_6) = CAS(input(6), input(7)) // Swap order

  // === Stage 2 ===
  val (s2_0_t, s2_2_t) = CAS(s1_0, s1_2)
  val (s2_1_t, s2_3_t) = CAS(s1_1, s1_3)
  val (s2_0, s2_1) = CAS(s2_0_t, s2_1_t)
  val (s2_2, s2_3) = CAS(s2_2_t, s2_3_t)

  val (s2_6_t, s2_4_t) = CAS(s1_4, s1_6) // Descending merge
  val (s2_7_t, s2_5_t) = CAS(s1_5, s1_7)
  val (s2_5, s2_4) = CAS(s2_4_t, s2_5_t)
  val (s2_7, s2_6) = CAS(s2_6_t, s2_7_t)

  // === Stage 3 ===
  val (s3_0_t1, s3_4_t1) = CAS(s2_0, s2_4)
  val (s3_1_t1, s3_5_t1) = CAS(s2_1, s2_5)
  val (s3_2_t1, s3_6_t1) = CAS(s2_2, s2_6)
  val (s3_3_t1, s3_7_t1) = CAS(s2_3, s2_7)

  val (s3_0_t2, s3_2_t2) = CAS(s3_0_t1, s3_2_t1)
  val (s3_1_t2, s3_3_t2) = CAS(s3_1_t1, s3_3_t1)
  val (s3_4_t2, s3_6_t2) = CAS(s3_4_t1, s3_6_t1)
  val (s3_5_t2, s3_7_t2) = CAS(s3_5_t1, s3_7_t1)

  val (o0, o1) = CAS(s3_0_t2, s3_1_t2)
  val (o2, o3) = CAS(s3_2_t2, s3_3_t2)
  val (o4, o5) = CAS(s3_4_t2, s3_5_t2)
  val (o6, o7) = CAS(s3_6_t2, s3_7_t2)

  output(0) := o0
  output(1) := o1
  output(2) := o2
  output(3) := o3
  output(4) := o4
  output(5) := o5
  output(6) := o6
  output(7) := o7
}

object Sort extends App {
  (new ChiselStage).execute(args, Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new Sort(8))))
}

