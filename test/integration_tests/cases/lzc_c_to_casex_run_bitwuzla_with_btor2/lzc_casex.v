module lzc7_casex
(
    input   [7:0]   mant_in,
    output  [7:0]   out_0
);

reg [2:0] lzc_not_special;
always @(*) begin
    casex(mant_in)
        8'b1xxxxxxx : lzc_not_special = 3'd0;
        8'b01xxxxxx : lzc_not_special = 3'd1;
        8'b001xxxxx : lzc_not_special = 3'd2;
        8'b0001xxxx : lzc_not_special = 3'd3;
        8'b00001xxx : lzc_not_special = 3'd4;
        8'b000001xx : lzc_not_special = 3'd5;
	8'b0000001x : lzc_not_special = 3'd6;
        default    : lzc_not_special = 3'd7;
    endcase
end

assign out_0            = {5'h0, lzc_not_special};

endmodule
