module kan_processing_core #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 16,
    parameter NUM_COEFFS = 64,
    parameter NUM_KNOTS = 32,
    parameter PIPELINE_DEPTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] x_input,
    input wire x_valid,
    output wire x_ready,
    
    output wire [DATA_WIDTH-1:0] y_output,
    output wire y_valid,
    input wire y_ready,
    
    input wire [ADDR_WIDTH-1:0] coeff_addr,
    input wire [DATA_WIDTH-1:0] coeff_data,
    input wire coeff_wr_en,
    
    input wire [ADDR_WIDTH-1:0] knot_addr,
    input wire [DATA_WIDTH-1:0] knot_data,
    input wire knot_wr_en,
    
    output wire busy,
    output wire error
);

wire [DATA_WIDTH-1:0] coefficient_mem [NUM_COEFFS-1:0];
wire [DATA_WIDTH-1:0] knot_vector [NUM_KNOTS-1:0];

reg [DATA_WIDTH-1:0] pipeline_reg [PIPELINE_DEPTH-1:0];
reg [PIPELINE_DEPTH-1:0] valid_pipeline;

wire [DATA_WIDTH-1:0] bspline_basis [NUM_KNOTS-2:0];
wire [DATA_WIDTH-1:0] weighted_sum;

reg [5:0] coeff_idx;
reg [5:0] knot_idx;
reg computation_active;

memory_bank #(
    .DEPTH(NUM_COEFFS),
    .WIDTH(DATA_WIDTH)
) coeff_memory (
    .clk(clk),
    .rst_n(rst_n),
    .addr(coeff_addr),
    .data_in(coeff_data),
    .wr_en(coeff_wr_en),
    .data_out(coefficient_mem[coeff_addr])
);

memory_bank #(
    .DEPTH(NUM_KNOTS),
    .WIDTH(DATA_WIDTH)
) knot_memory (
    .clk(clk),
    .rst_n(rst_n),
    .addr(knot_addr),
    .data_in(knot_data),
    .wr_en(knot_wr_en),
    .data_out(knot_vector[knot_addr])
);

genvar i;
generate
    for (i = 0; i < NUM_KNOTS-1; i = i + 1) begin : bspline_units
        bspline_evaluator #(
            .DATA_WIDTH(DATA_WIDTH)
        ) bspline_eval (
            .clk(clk),
            .rst_n(rst_n),
            .x(pipeline_reg[0]),
            .knot_left(knot_vector[i]),
            .knot_right(knot_vector[i+1]),
            .basis_value(bspline_basis[i])
        );
    end
endgenerate

multiply_accumulate #(
    .DATA_WIDTH(DATA_WIDTH),
    .NUM_INPUTS(NUM_KNOTS-1)
) mac_unit (
    .clk(clk),
    .rst_n(rst_n),
    .coefficients(coefficient_mem),
    .basis_functions(bspline_basis),
    .result(weighted_sum)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pipeline_reg <= '{default: '0};
        valid_pipeline <= '0;
        computation_active <= 1'b0;
        coeff_idx <= '0;
        knot_idx <= '0;
    end else if (enable) begin
        if (x_valid && x_ready) begin
            pipeline_reg[0] <= x_input;
            valid_pipeline[0] <= 1'b1;
            computation_active <= 1'b1;
        end
        
        for (int j = 1; j < PIPELINE_DEPTH; j = j + 1) begin
            pipeline_reg[j] <= pipeline_reg[j-1];
            valid_pipeline[j] <= valid_pipeline[j-1];
        end
        
        if (computation_active) begin
            coeff_idx <= coeff_idx + 1;
            if (coeff_idx == NUM_COEFFS-1) begin
                coeff_idx <= '0;
                knot_idx <= knot_idx + 1;
                if (knot_idx == NUM_KNOTS-1) begin
                    knot_idx <= '0;
                    computation_active <= 1'b0;
                end
            end
        end
    end
end

assign x_ready = !computation_active && enable;
assign y_output = weighted_sum;
assign y_valid = valid_pipeline[PIPELINE_DEPTH-1] && !computation_active;
assign busy = computation_active;
assign error = 1'b0;

endmodule

module memory_bank #(
    parameter DEPTH = 64,
    parameter WIDTH = 32,
    parameter ADDR_WIDTH = $clog2(DEPTH)
)(
    input wire clk,
    input wire rst_n,
    input wire [ADDR_WIDTH-1:0] addr,
    input wire [WIDTH-1:0] data_in,
    input wire wr_en,
    output reg [WIDTH-1:0] data_out
);

reg [WIDTH-1:0] memory [DEPTH-1:0];

always @(posedge clk) begin
    if (wr_en) begin
        memory[addr] <= data_in;
    end
    data_out <= memory[addr];
end

endmodule

module bspline_evaluator #(
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] x,
    input wire [DATA_WIDTH-1:0] knot_left,
    input wire [DATA_WIDTH-1:0] knot_right,
    output reg [DATA_WIDTH-1:0] basis_value
);

wire [DATA_WIDTH-1:0] knot_diff;
wire [DATA_WIDTH-1:0] x_normalized;
wire [DATA_WIDTH-1:0] basis_calc;

assign knot_diff = knot_right - knot_left;
assign x_normalized = (x >= knot_left && x <= knot_right) ? (x - knot_left) : '0;

fp_divider #(
    .DATA_WIDTH(DATA_WIDTH)
) divider (
    .clk(clk),
    .rst_n(rst_n),
    .dividend(x_normalized),
    .divisor(knot_diff),
    .quotient(basis_calc)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        basis_value <= '0;
    end else begin
        if (x >= knot_left && x <= knot_right && knot_diff != 0) begin
            basis_value <= basis_calc;
        end else begin
            basis_value <= '0;
        end
    end
end

endmodule

module multiply_accumulate #(
    parameter DATA_WIDTH = 32,
    parameter NUM_INPUTS = 31
)(
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] coefficients [NUM_INPUTS-1:0],
    input wire [DATA_WIDTH-1:0] basis_functions [NUM_INPUTS-1:0],
    output reg [DATA_WIDTH-1:0] result
);

reg [DATA_WIDTH-1:0] products [NUM_INPUTS-1:0];
reg [DATA_WIDTH-1:0] accumulator;

genvar i;
generate
    for (i = 0; i < NUM_INPUTS; i = i + 1) begin : mult_units
        fp_multiplier #(
            .DATA_WIDTH(DATA_WIDTH)
        ) multiplier (
            .clk(clk),
            .rst_n(rst_n),
            .a(coefficients[i]),
            .b(basis_functions[i]),
            .product(products[i])
        );
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        accumulator <= '0;
        result <= '0;
    end else begin
        accumulator = '0;
        for (int j = 0; j < NUM_INPUTS; j = j + 1) begin
            accumulator = accumulator + products[j];
        end
        result <= accumulator;
    end
end

endmodule

module fp_multiplier #(
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] a,
    input wire [DATA_WIDTH-1:0] b,
    output reg [DATA_WIDTH-1:0] product
);

reg [2*DATA_WIDTH-1:0] temp_product;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        product <= '0;
        temp_product <= '0;
    end else begin
        temp_product = a * b;
        product <= temp_product[DATA_WIDTH-1:0];
    end
end

endmodule

module fp_divider #(
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] dividend,
    input wire [DATA_WIDTH-1:0] divisor,
    output reg [DATA_WIDTH-1:0] quotient
);

reg [DATA_WIDTH-1:0] temp_quotient;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        quotient <= '0;
    end else begin
        if (divisor != 0) begin
            temp_quotient = dividend / divisor;
            quotient <= temp_quotient;
        end else begin
            quotient <= '0;
        end
    end
end

endmodule 