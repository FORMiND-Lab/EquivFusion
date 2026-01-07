module top (input [7:0] a,
	    input [3:0] b,
	    output [15:0] sum,
	    output reg [7:0] out);
    
    assign sum = a + b;

    always_comb begin
	if (a > b) begin
	    out = a + 1;
	end else begin 
	    out = b + 1;
	end
    end

endmodule
