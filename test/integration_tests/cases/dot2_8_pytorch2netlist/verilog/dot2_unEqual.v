module dot2_comb #(
    parameter N       = 2,   // Vector length
    parameter W       = 8,   // Bit-width of each element (signed)
    parameter ACC_W   = 32,    // Bit-width of accumulator
    parameter O_W = 32          // Bit-width of Output
)(
    // Two signed input vectors, each containing N elements of W bits
    input  logic signed [N-1:0] [W-1:0] arg_0,
    input  logic signed [N-1:0] [W-1:0] arg_1,

    // Signed output: dot product result (sum of element-wise products)
    output logic signed [0:0][O_W-1:0] out_0
);

    //============================================================
    // 1. Element-wise multiplications
    // Each pair a[i], b[i] is multiplied to produce a 2*W-bit product.
    //============================================================
    logic signed [N-1:0] [(2*W)-1:0] prod;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : GEN_PROD
            assign prod[i] = $signed(arg_0[i]) * $signed(arg_1[i]);
        end
    endgenerate

    //============================================================
    // 2. Summation of all products
    // This always block performs a combinational reduction (sum)
    // across all products. The result is stored in sum_reg.
    //============================================================
    integer k;
    logic signed [ACC_W-1:0] sum_reg;

    always @(*) begin
        sum_reg = '0; // Initialize accumulator to zero
        for (k = 0; k < N; k = k + 1) begin
            // Sign-extend each 2*W-bit product to ACC_W bits
            // before adding, to prevent overflow and preserve sign.
            sum_reg = sum_reg +
                {{(ACC_W-(2*W)){prod[k][(2*W)-1]}}, prod[k]};
        end
    end

    //============================================================
    // 3. Final output assignment
    // The dot product is simply the total accumulated sum.
    //============================================================
    assign out_0 = sum_reg[O_W-1:0];

endmodule
