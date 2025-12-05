module array_assign(
    output reg o_1[1:0],
    input in_2,       output reg o_2[1:0],
    input in_3,       output reg [1:0] o_3,
    input in_4 [1:0], output reg o_4,
    input in_5,       output reg o_5[1:0],
    input in_6[2:0],  output reg o_6[1:0][1:0]
);
    wire temp[1:0] = '{1'b1, 1'b0};

    assign o_1 = temp;                      // hw.aggregate_constant
    assign o_2 = {in_2, in_2};              // comb.replicate + hw.bitcast
    assign o_3 = {in_3, in_3};              // comb.replicate
    assign o_4 = in_4[1] & in_4[0];         // hw.array_get

    assign o_5[0] = in_5;                   // hw.array_create
    assign o_5[1] = in_5;

    assign o_6[1] = in_6[1:0];              // hw.array_slice
    assign o_6[0] = in_6[1:0];
endmodule