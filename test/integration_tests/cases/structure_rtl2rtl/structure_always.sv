typedef struct packed {int a; int b;} StructType;

module structure(
    input int in_int,
    input StructType in_struct,
    output StructType out_struct_1,
    output StructType out_struct_2,
    output StructType out_struct_3,
    output StructType out_struct_4,
    output int out_int,
    output StructType out_struct_5
);
    always_comb begin
        out_struct_1 = in_struct;

        out_struct_2 = {32'b0, 32'b1};
        out_struct_3 = '{32'b0, 32'b1};

        out_struct_4 = '{in_int, in_int};           // hw.struct_create

        out_int = in_struct.a;                      // hw.struct_extract

        out_struct_5.a = in_int;                    // hw.bitcast + hw.struct_create + hw.struct_extract
        out_struct_5.b = in_int;
    end
endmodule