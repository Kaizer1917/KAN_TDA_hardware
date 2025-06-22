module kan_processing_element #(
    parameter DATA_WIDTH = 32,
    parameter COEFF_WIDTH = 24,
    parameter NUM_SPLINES = 16,
    parameter PIPELINE_STAGES = 4,
    parameter LUT_DEPTH = 256
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
    
    input wire [4:0] spline_select,
    input wire [COEFF_WIDTH-1:0] coefficients [NUM_SPLINES-1:0],
    input wire coeff_valid,
    
    output wire busy,
    output wire overflow_flag
);

reg [DATA_WIDTH-1:0] pipeline_x [PIPELINE_STAGES-1:0];
reg [PIPELINE_STAGES-1:0] pipeline_valid;
reg [DATA_WIDTH-1:0] normalized_x;

wire [DATA_WIDTH-1:0] basis_functions [NUM_SPLINES-1:0];
wire [DATA_WIDTH-1:0] weighted_outputs [NUM_SPLINES-1:0];
wire [DATA_WIDTH-1:0] final_sum;

reg [7:0] lut_addr;
wire [DATA_WIDTH-1:0] lut_output;

bspline_lut #(
    .DATA_WIDTH(DATA_WIDTH),
    .LUT_DEPTH(LUT_DEPTH)
) spline_lut (
    .clk(clk),
    .rst_n(rst_n),
    .addr(lut_addr),
    .data_out(lut_output)
);

genvar i;
generate
    for (i = 0; i < NUM_SPLINES; i = i + 1) begin : spline_units
        spline_basis_function #(
            .DATA_WIDTH(DATA_WIDTH),
            .COEFF_WIDTH(COEFF_WIDTH)
        ) basis_func (
            .clk(clk),
            .rst_n(rst_n),
            .x(normalized_x),
            .coefficient(coefficients[i]),
            .basis_value(basis_functions[i])
        );
        
        multiplier_dsp #(
            .DATA_WIDTH(DATA_WIDTH),
            .COEFF_WIDTH(COEFF_WIDTH)
        ) weight_mult (
            .clk(clk),
            .rst_n(rst_n),
            .a(basis_functions[i]),
            .b(coefficients[i]),
            .product(weighted_outputs[i])
        );
    end
endgenerate

tree_adder #(
    .DATA_WIDTH(DATA_WIDTH),
    .NUM_INPUTS(NUM_SPLINES)
) adder_tree (
    .clk(clk),
    .rst_n(rst_n),
    .inputs(weighted_outputs),
    .sum(final_sum),
    .overflow(overflow_flag)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pipeline_x <= '{default: '0};
        pipeline_valid <= '0;
        normalized_x <= '0;
        lut_addr <= 8'h0;
    end else if (enable) begin
        if (x_valid && x_ready) begin
            pipeline_x[0] <= x_input;
            pipeline_valid[0] <= 1'b1;
            normalized_x <= x_input;
            lut_addr <= x_input[7:0];
        end
        
        for (int j = 1; j < PIPELINE_STAGES; j = j + 1) begin
            pipeline_x[j] <= pipeline_x[j-1];
            pipeline_valid[j] <= pipeline_valid[j-1];
        end
    end
end

assign x_ready = !busy && enable;
assign y_output = final_sum;
assign y_valid = pipeline_valid[PIPELINE_STAGES-1] && coeff_valid;
assign busy = pipeline_valid[0] || pipeline_valid[1];

endmodule

module spline_basis_function #(
    parameter DATA_WIDTH = 32,
    parameter COEFF_WIDTH = 24
)(
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] x,
    input wire [COEFF_WIDTH-1:0] coefficient,
    output reg [DATA_WIDTH-1:0] basis_value
);

reg [DATA_WIDTH-1:0] x_squared;
reg [DATA_WIDTH-1:0] x_cubed;
reg [DATA_WIDTH-1:0] basis_calc;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        x_squared <= '0;
        x_cubed <= '0;
        basis_calc <= '0;
        basis_value <= '0;
    end else begin
        x_squared <= x * x;
        x_cubed <= x_squared * x;
        
        if (x >= 0 && x < 32'h40000000) begin
            basis_calc <= x_cubed - 3 * x_squared + 3 * x - 1;
        end else if (x >= 32'h40000000 && x < 32'h80000000) begin
            basis_calc <= -3 * x_cubed + 12 * x_squared - 12 * x + 4;
        end else if (x >= 32'h80000000 && x < 32'hC0000000) begin
            basis_calc <= 3 * x_cubed - 24 * x_squared + 60 * x - 44;
        end else begin
            basis_calc <= '0;
        end
        
        basis_value <= (basis_calc * coefficient) >> 16;
    end
end

endmodule

module multiplier_dsp #(
    parameter DATA_WIDTH = 32,
    parameter COEFF_WIDTH = 24
)(
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] a,
    input wire [COEFF_WIDTH-1:0] b,
    output reg [DATA_WIDTH-1:0] product
);

reg [DATA_WIDTH+COEFF_WIDTH-1:0] temp_product;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        temp_product <= '0;
        product <= '0;
    end else begin
        temp_product <= a * b;
        product <= temp_product[DATA_WIDTH+COEFF_WIDTH-1:COEFF_WIDTH];
    end
end

endmodule

module tree_adder #(
    parameter DATA_WIDTH = 32,
    parameter NUM_INPUTS = 16
)(
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] inputs [NUM_INPUTS-1:0],
    output reg [DATA_WIDTH-1:0] sum,
    output reg overflow
);

reg [DATA_WIDTH-1:0] level1_sums [NUM_INPUTS/2-1:0];
reg [DATA_WIDTH-1:0] level2_sums [NUM_INPUTS/4-1:0];
reg [DATA_WIDTH-1:0] level3_sums [NUM_INPUTS/8-1:0];
reg [DATA_WIDTH-1:0] level4_sums [NUM_INPUTS/16-1:0];

genvar i;
generate
    for (i = 0; i < NUM_INPUTS/2; i = i + 1) begin : level1_add
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                level1_sums[i] <= '0;
            end else begin
                level1_sums[i] <= inputs[2*i] + inputs[2*i+1];
            end
        end
    end
    
    for (i = 0; i < NUM_INPUTS/4; i = i + 1) begin : level2_add
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                level2_sums[i] <= '0;
            end else begin
                level2_sums[i] <= level1_sums[2*i] + level1_sums[2*i+1];
            end
        end
    end
    
    for (i = 0; i < NUM_INPUTS/8; i = i + 1) begin : level3_add
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                level3_sums[i] <= '0;
            end else begin
                level3_sums[i] <= level2_sums[2*i] + level2_sums[2*i+1];
            end
        end
    end
    
    if (NUM_INPUTS >= 16) begin : level4_add
        for (i = 0; i < NUM_INPUTS/16; i = i + 1) begin
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    level4_sums[i] <= '0;
                end else begin
                    level4_sums[i] <= level3_sums[2*i] + level3_sums[2*i+1];
                end
            end
        end
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sum <= '0;
        overflow <= 1'b0;
    end else begin
        if (NUM_INPUTS == 16) begin
            sum <= level4_sums[0];
        end else if (NUM_INPUTS == 8) begin
            sum <= level3_sums[0];
        end else if (NUM_INPUTS == 4) begin
            sum <= level2_sums[0];
        end else if (NUM_INPUTS == 2) begin
            sum <= level1_sums[0];
        end else begin
            sum <= inputs[0];
        end
        
        overflow <= (sum > {1'b0, {DATA_WIDTH-1{1'b1}}});
    end
end

endmodule

module bspline_lut #(
    parameter DATA_WIDTH = 32,
    parameter LUT_DEPTH = 256
)(
    input wire clk,
    input wire rst_n,
    input wire [7:0] addr,
    output reg [DATA_WIDTH-1:0] data_out
);

reg [DATA_WIDTH-1:0] lut_memory [LUT_DEPTH-1:0];

initial begin
    for (int i = 0; i < LUT_DEPTH; i = i + 1) begin
        lut_memory[i] = $sin(i * 3.14159 / 128) * (1 << 24);
    end
end

always @(posedge clk) begin
    if (rst_n) begin
        data_out <= lut_memory[addr];
    end else begin
        data_out <= '0;
    end
end

endmodule

module kan_array_controller #(
    parameter NUM_ELEMENTS = 16,
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] input_stream,
    input wire input_valid,
    output wire input_ready,
    
    output wire [DATA_WIDTH-1:0] output_stream,
    output wire output_valid,
    input wire output_ready,
    
    output wire [NUM_ELEMENTS-1:0] element_enable,
    output wire [4:0] element_select [NUM_ELEMENTS-1:0],
    
    input wire [NUM_ELEMENTS-1:0] element_busy,
    input wire [NUM_ELEMENTS-1:0] element_overflow,
    
    output wire array_ready,
    output wire processing_error
);

reg [3:0] state;
reg [7:0] element_counter;
reg [DATA_WIDTH-1:0] data_buffer [NUM_ELEMENTS-1:0];
reg [NUM_ELEMENTS-1:0] buffer_valid;

localparam IDLE = 4'h0;
localparam LOADING = 4'h1;
localparam PROCESSING = 4'h2;
localparam UNLOADING = 4'h3;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        element_counter <= 8'h0;
        buffer_valid <= '0;
        for (int i = 0; i < NUM_ELEMENTS; i = i + 1) begin
            data_buffer[i] <= '0;
        end
    end else if (enable) begin
        case (state)
            IDLE: begin
                if (input_valid) begin
                    state <= LOADING;
                    element_counter <= 8'h0;
                end
            end
            
            LOADING: begin
                if (input_valid && element_counter < NUM_ELEMENTS) begin
                    data_buffer[element_counter] <= input_stream;
                    buffer_valid[element_counter] <= 1'b1;
                    element_counter <= element_counter + 1;
                    
                    if (element_counter == NUM_ELEMENTS - 1) begin
                        state <= PROCESSING;
                        element_counter <= 8'h0;
                    end
                end
            end
            
            PROCESSING: begin
                if (!|element_busy) begin
                    state <= UNLOADING;
                    element_counter <= 8'h0;
                end
            end
            
            UNLOADING: begin
                if (output_ready && element_counter < NUM_ELEMENTS) begin
                    element_counter <= element_counter + 1;
                    buffer_valid[element_counter] <= 1'b0;
                    
                    if (element_counter == NUM_ELEMENTS - 1) begin
                        state <= IDLE;
                        element_counter <= 8'h0;
                    end
                end
            end
        endcase
    end
end

assign input_ready = (state == IDLE) || (state == LOADING && element_counter < NUM_ELEMENTS);
assign output_valid = (state == UNLOADING);
assign output_stream = data_buffer[element_counter];
assign element_enable = buffer_valid;
assign array_ready = (state == IDLE);
assign processing_error = |element_overflow;

genvar j;
generate
    for (j = 0; j < NUM_ELEMENTS; j = j + 1) begin : element_selects
        assign element_select[j] = j[4:0];
    end
endgenerate

endmodule 