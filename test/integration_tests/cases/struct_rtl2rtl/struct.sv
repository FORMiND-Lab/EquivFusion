typedef struct packed {int a; int b;} StructType;

module top(
    input int in_int,
    input StructType in_struct_1,
    input StructType in_struct_2,
    output StructType out_struct_1,
    output StructType out_struct_2,
    output StructType out_struct_3,
    output StructType out_struct_4,
    output StructType out_struct_5,
    output StructType out_struct_6,
    output StructType out_struct_7,
    output int out_int_1
);
    assign out_struct_1 = in_struct_1;

    assign out_struct_2 = '{32'b0, 32'b1};                              // %0 = hw.aggregate_constant [0 : i32, 1 : i32] : !hw.struct<a: i32, b: i32>
    assign out_struct_3 = '{in_int, in_int};                            // %1 = hw.struct_create (%in_int, %in_int) : !hw.struct<a: i32, b: i32>

    assign out_struct_4 = {32'b0, 32'b1};                               // %2 = hw.bitcast %c1_i64 : (i64) -> !hw.struct<a: i32, b: i32>

    assign out_struct_5.a = in_int;
    assign out_struct_5.b = in_int;                                     // %1

    assign out_int_1 = in_struct_1.a;                                   // %a = hw.struct_extract %in_struct_1["a"] : !hw.struct<a: i32, b: i32>

    always_comb begin
        out_struct_6.a = in_int;                                        //  %3 = hw.struct_inject %3["a"], %in_int : !hw.struct<a: i32, b: i32>
    end

    assign out_struct_7 = in_int ? in_struct_1 : in_struct_2;           //  %4 = comb.icmp ne %in_int, %c0_i32 : i32
                                                                        //  %5 = comb.mux %4, %in_struct_1, %in_struct_2 : !hw.struct<a: i32, b: i32>

endmodule

/*
// circt-verilog struct.sv -o struct.mlir
module {
  hw.module @top(in %in_int : i32, in %in_struct_1 : !hw.struct<a: i32, b: i32>, in %in_struct_2 : !hw.struct<a: i32, b: i32>, out out_struct_1 : !hw.struct<a: i32, b: i32>, out out_struct_2 : !hw.struct<a: i32, b: i32>, out out_struct_3 : !hw.struct<a: i32, b: i32>, out out_struct_4 : !hw.struct<a: i32, b: i32>, out out_struct_5 : !hw.struct<a: i32, b: i32>, out out_struct_6 : !hw.struct<a: i32, b: i32>, out out_struct_7 : !hw.struct<a: i32, b: i32>, out out_int_1 : i32) {
    %c1_i64 = hw.constant 1 : i64
    %0 = hw.aggregate_constant [0 : i32, 1 : i32] : !hw.struct<a: i32, b: i32>
    %c0_i32 = hw.constant 0 : i32
    %1 = hw.struct_create (%in_int, %in_int) : !hw.struct<a: i32, b: i32>
    %2 = hw.bitcast %c1_i64 : (i64) -> !hw.struct<a: i32, b: i32>
    %a = hw.struct_extract %in_struct_1["a"] : !hw.struct<a: i32, b: i32>
    %3 = hw.struct_inject %3["a"], %in_int : !hw.struct<a: i32, b: i32>
    %4 = comb.icmp ne %in_int, %c0_i32 : i32
    %5 = comb.mux %4, %in_struct_1, %in_struct_2 : !hw.struct<a: i32, b: i32>
    hw.output %in_struct_1, %0, %1, %2, %1, %3, %5, %a : !hw.struct<a: i32, b: i32>, !hw.struct<a: i32, b: i32>, !hw.struct<a: i32, b: i32>, !hw.struct<a: i32, b: i32>, !hw.struct<a: i32, b: i32>, !hw.struct<a: i32, b: i32>, !hw.struct<a: i32, b: i32>, i32
  }
}
*/