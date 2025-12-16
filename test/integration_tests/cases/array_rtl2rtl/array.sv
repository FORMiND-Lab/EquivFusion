module top(
    input logic in_logic_1,
    input in_array_1 [1:0],
    input in_array_2 [1:0],
    input in_array_3 [2:0],

    output [1:0] out_array_0,

    output out_array_1 [1:0],
    output out_array_2 [1:0],
    output out_array_3 [1:0],
    output out_array_4 [1:0],
    output out_array_5 [1:0],
    output out_array_6 [1:0],
    output reg out_array_7 [1:0],
    output out_array_8 [1:0],

    output logic out_logic_1
);
    assign out_array_1 = in_array_1;

    assign out_array_2 = '{1'b1, 1'b0};                                 //  %0 = hw.aggregate_constant [true, false] : !hw.array<2xi1>
    assign out_array_3 = '{in_logic_1, in_logic_1};                     //  %1 = hw.array_create %in_logic_1, %in_logic_1 : i1

    assign out_array_0 = {in_logic_1, in_logic_1};                      //  %2 = comb.replicate %in_logic_1 : (i1) -> i2
    assign out_array_4 = {1'b1, 1'b0};                                  //  %3 = hw.bitcast %c-2_i2 : (i2) -> !hw.array<2xi1>

    assign out_array_5[0] = in_logic_1;
    assign out_array_5[1] = in_logic_1;                                 //  %1

    assign out_logic_1 = in_array_1[1];                                 //  %4 = hw.array_get %in_array_1[%true] : !hw.array<2xi1>, i1

    assign out_array_6 = in_array_3[1:0];                               //  %5 = hw.array_slice %in_array_3[%c0_i2] : (!hw.array<3xi1>) -> !hw.array<2xi1>

    always_comb begin
        out_array_7[1] = in_logic_1;                                    //  %6 = hw.array_inject %6[%true], %in_logic_1 : !hw.array<2xi1>, i1
    end

    assign out_array_8 = in_logic_1 ? in_array_1 : in_array_2;         //   %7 = comb.mux %in_logic_1, %in_array_1, %in_array_2 : !hw.array<2xi1>
endmodule


/*
// circt-verilog array.sv -o array.mlir
module {
  hw.module @array_assign(in %in_logic_1 : i1, in %in_array_1 : !hw.array<2xi1>, in %in_array_2 : !hw.array<2xi1>, in %in_array_3 : !hw.array<3xi1>, out out_array_0 : i2, out out_array_1 : !hw.array<2xi1>, out out_array_2 : !hw.array<2xi1>, out out_array_3 : !hw.array<2xi1>, out out_array_4 : !hw.array<2xi1>, out out_array_5 : !hw.array<2xi1>, out out_array_6 : !hw.array<2xi1>, out out_array_7 : !hw.array<2xi1>, out out_array_8 : !hw.array<2xi1>, out out_logic_1 : i1) {
    %c-2_i2 = hw.constant -2 : i2
    %0 = hw.aggregate_constant [true, false] : !hw.array<2xi1>
    %c0_i2 = hw.constant 0 : i2
    %true = hw.constant true
    %1 = hw.array_create %in_logic_1, %in_logic_1 : i1
    %2 = comb.replicate %in_logic_1 : (i1) -> i2
    %3 = hw.bitcast %c-2_i2 : (i2) -> !hw.array<2xi1>
    %4 = hw.array_get %in_array_1[%true] : !hw.array<2xi1>, i1
    %5 = hw.array_slice %in_array_3[%c0_i2] : (!hw.array<3xi1>) -> !hw.array<2xi1>
    %6 = hw.array_inject %6[%true], %in_logic_1 : !hw.array<2xi1>, i1
    %7 = comb.mux %in_logic_1, %in_array_1, %in_array_2 : !hw.array<2xi1>
    hw.output %2, %in_array_1, %0, %1, %3, %1, %5, %6, %7, %4 : i2, !hw.array<2xi1>, !hw.array<2xi1>, !hw.array<2xi1>, !hw.array<2xi1>, !hw.array<2xi1>, !hw.array<2xi1>, !hw.array<2xi1>, !hw.array<2xi1>, i1
  }
}

*/