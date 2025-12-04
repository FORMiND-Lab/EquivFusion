module array_always(
    output reg o_1[1:0],
    input in_2,       output reg o_2[1:0],
    input in_3,       output reg [1:0] o_3,
    input in_4 [1:0], output reg o_4,
    input in_5,       output reg o_5[1:0],
    input in_6[2:0],  output reg o_6[1:0][1:0]
);
    wire temp[1:0] = '{1'b1, 1'b0};

    always @(*) begin
        o_1 = temp;
        o_2 = {in_2, in_2};
        o_3 = {in_3, in_3};
        o_4 = in_4[1] & in_4[0];
        
        o_5[0] = in_5;
        o_5[1] = in_5;

        o_6[1] = in_6[1-:2];
        o_6[0] = in_6[0+:2];
    end
endmodule
            
