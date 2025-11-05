
module lod4
(
    input   [3:0]   x,
    output  [3:0]   y,     // one hot code
    output          zero   // all bits are zero
);

assign y[3] = x[3];
assign y[2] = ~x[3] & x[2];
assign y[1] = ~x[3] & ~x[2] & x[1];
assign y[0] = ~x[3] & ~x[2] & ~x[1] & x[0];
assign zero = ~|x;

endmodule


module lod8
(
    input   [7:0]   x,
    output  [7:0]   y,
    output          zero
);

wire [3:0] y_hi, y_lo;
wire       z_hi, z_lo;

lod4 u_hi (.x(x[7:4]), .y(y_hi), .zero(z_hi));
lod4 u_lo (.x(x[3:0]), .y(y_lo), .zero(z_lo));

assign y    = z_hi ? {4'b0, y_lo} : {y_hi, 4'b0};
assign zero = z_hi & z_lo;

endmodule


module lzc7_loop
(
    input   [7:0]   mant_in,
    output  [7:0]   out_0
);

wire [2:0] count   [8:0];
wire [7:0] data_one_hot;
wire [7:0] data_zero_two_one;
wire is_zero;

lod8 u_lod8
(
    .x(mant_in),
    .y(data_one_hot),
    .zero(is_zero)
);

assign data_zero_two_one = (~data_one_hot[7:0] + 1'b1) ^ data_one_hot[7:0];

genvar i;
assign count[8] = 3'b0;
generate
    for (i = 1; i < 9; i = i+1) begin: countaaa
        assign count[8-i] = data_zero_two_one[8-i] ? count[8-i+1] + 1'b1 : count[8-i+1];
    end
endgenerate

assign out_0            = {5'b0,  is_zero ? 3'b111 : count[0]};

endmodule


