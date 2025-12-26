import chisel3._
import circt.stage.ChiselStage
import java.io._

class Adder extends RawModule {
  val a = IO(Input(UInt(8.W)))
  val b = IO(Input(UInt(8.W)))
  val sum = IO(Output(UInt(8.W)))
  sum := a + b
}

object Adder extends App {
  // 将所有命令行参数 args 直接传递给 ChiselStage.execute
  // ChiselStage 会自动解析参数，并根据参数执行操作（生成FIRRTL或Verilog）
  (new ChiselStage).execute(args, Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new Adder)))
}
