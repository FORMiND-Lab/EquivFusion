module top (input [3:0] b,
	    input [7:0] a,
	    output reg [7:0] out,
	    output [15:0] sum);
    
    assign sum = a + b;

    always_comb begin
	if (a > b) begin
	    out = a + 1;
	end else begin 
	    out = b + 1;
	end
    end

endmodule
