module lzc7_dichotomy
(
    input   [7:0]   mant_in,
        
    output  [7:0]   out_0                                
);

wire [2:0] leading_one ;

assign leading_one[2] = |mant_in[7:4] ;
assign leading_one[1] = (|mant_in[7:4]) ? (|mant_in[7:6]) : (|mant_in[3:2]) ;
assign leading_one[0] = (|mant_in[7:4]) ? (|mant_in[7:6]) ? mant_in[7] : mant_in[5] : (|mant_in[3:2]) ? mant_in[3] : mant_in[1] ;
//(|mant_in[6:3]) ? (mant_in[6] | (~mant_in[5] & mant_in[4])) : (mant_in[2] | (~mant_in[1] & mant_in[0])) ;

assign out_0 = {5'h0, ~leading_one};

endmodule
