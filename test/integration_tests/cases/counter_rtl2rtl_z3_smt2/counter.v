module counter (
    input wire clk,        // 时钟信号
    input wire rst_n,      // 复位信号（低有效）
    output reg [3:0] count // 4位计数输出
);

// 时序逻辑：每个时钟上升沿计数
always @(posedge clk) begin
    if (!rst_n) 
        count <= 4'b0000;  // 复位时清零
    else 
        count <= count + 4'b0001;  // 否则计数加1
end

endmodule
