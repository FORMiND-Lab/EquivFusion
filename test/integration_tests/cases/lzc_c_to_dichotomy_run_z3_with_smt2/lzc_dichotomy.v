module lzc7_dichotomy
(
    input   [7:0]   in_0,
        
    output  [7:0]   out_0                                
);

wire [2:0] leading_one ;

assign leading_one[2] = |in_0[7:4] ;
assign leading_one[1] = (|in_0[7:4]) ? (|in_0[7:6]) : (|in_0[3:2]) ;
assign leading_one[0] = (|in_0[7:4]) ? (|in_0[7:6]) ? in_0[7] : in_0[5] : (|in_0[3:2]) ? in_0[3] : in_0[1] ;
//(|in_0[6:3]) ? (in_0[6] | (~in_0[5] & in_0[4])) : (in_0[2] | (~in_0[1] & in_0[0])) ;

assign out_0 = {5'h0, ~leading_one};

endmodule
