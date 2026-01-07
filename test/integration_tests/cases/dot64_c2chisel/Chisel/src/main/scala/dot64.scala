import chisel3._
import circt.stage.ChiselStage

class Dot64 extends RawModule {
    val arg_0 = IO(Input(Vec(64, SInt(16.W))))
    val arg_1 = IO(Input(Vec(64, SInt(16.W))))
    val out_0 = IO(Output(SInt(64.W)))

    var sum = 0.S(64.W)
    for (i <- 0 until 64) {
        val product = arg_0(i) * arg_1(i)
        sum = sum + product
    }
    out_0 := sum
}

object Dot64 extends App {
  // 将所有命令行参数 args 直接传递给 ChiselStage.execute
  // ChiselStage 会自动解析参数，并根据参数执行操作（生成FIRRTL或Verilog）
  (new ChiselStage).execute(args, Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new Dot64)))
}
