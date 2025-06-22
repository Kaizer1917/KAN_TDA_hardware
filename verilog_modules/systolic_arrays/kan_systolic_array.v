module kan_systolic_array #(
    parameter DATA_WIDTH = 32,
    parameter ARRAY_SIZE = 8,
    parameter NUM_COEFFS = 16,
    parameter PIPELINE_DEPTH = 4
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] input_stream [ARRAY_SIZE-1:0],
    input wire [ARRAY_SIZE-1:0] input_valid,
    output wire [ARRAY_SIZE-1:0] input_ready,
    
    output wire [DATA_WIDTH-1:0] output_stream [ARRAY_SIZE-1:0],
    output wire [ARRAY_SIZE-1:0] output_valid,
    input wire [ARRAY_SIZE-1:0] output_ready,
    
    input wire [DATA_WIDTH-1:0] weights [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0],
    input wire weight_valid,
    
    output wire computation_active,
    output wire [7:0] throughput_counter
);

wire [DATA_WIDTH-1:0] pe_data_h [ARRAY_SIZE-1:0][ARRAY_SIZE:0];
wire [DATA_WIDTH-1:0] pe_data_v [ARRAY_SIZE:0][ARRAY_SIZE-1:0];
wire [ARRAY_SIZE-1:0] pe_valid_h [ARRAY_SIZE-1:0];
wire [ARRAY_SIZE-1:0] pe_valid_v [ARRAY_SIZE-1:0];

wire [DATA_WIDTH-1:0] partial_sums [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
wire [ARRAY_SIZE-1:0] pe_active [ARRAY_SIZE-1:0];

reg [7:0] cycle_counter;
reg [7:0] throughput_reg;

genvar i, j;
generate
    for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : array_rows
        for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : array_cols
            systolic_pe #(
                .DATA_WIDTH(DATA_WIDTH),
                .ROW_ID(i),
                .COL_ID(j)
            ) pe (
                .clk(clk),
                .rst_n(rst_n),
                .enable(enable),
                .data_in_h(pe_data_h[i][j]),
                .data_in_v(pe_data_v[i][j]),
                .weight(weights[i][j]),
                .data_out_h(pe_data_h[i][j+1]),
                .data_out_v(pe_data_v[i+1][j]),
                .partial_sum(partial_sums[i][j]),
                .valid_in_h(j == 0 ? input_valid[i] : pe_valid_h[i][j-1]),
                .valid_in_v(i == 0 ? 1'b1 : pe_valid_v[i-1][j]),
                .valid_out_h(pe_valid_h[i][j]),
                .valid_out_v(pe_valid_v[i][j]),
                .pe_active(pe_active[i][j])
            );
        end
    end
endgenerate

genvar k;
generate
    for (k = 0; k < ARRAY_SIZE; k = k + 1) begin : input_connections
        assign pe_data_h[k][0] = input_stream[k];
        assign pe_data_v[0][k] = input_stream[k];
        assign input_ready[k] = !pe_active[k][0] && enable;
    end
endgenerate

accumulator_array #(
    .DATA_WIDTH(DATA_WIDTH),
    .ARRAY_SIZE(ARRAY_SIZE)
) acc_array (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .partial_sums(partial_sums),
    .sum_valid(pe_active),
    .final_outputs(output_stream),
    .output_valid(output_valid)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cycle_counter <= 8'h0;
        throughput_reg <= 8'h0;
    end else if (enable) begin
        cycle_counter <= cycle_counter + 1;
        
        if (|output_valid) begin
            throughput_reg <= throughput_reg + 1;
        end
        
        if (cycle_counter == 8'hFF) begin
            cycle_counter <= 8'h0;
            throughput_reg <= 8'h0;
        end
    end
end

assign computation_active = |pe_active;
assign throughput_counter = throughput_reg;

endmodule

module systolic_pe #(
    parameter DATA_WIDTH = 32,
    parameter ROW_ID = 0,
    parameter COL_ID = 0
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] data_in_h,
    input wire [DATA_WIDTH-1:0] data_in_v,
    input wire [DATA_WIDTH-1:0] weight,
    
    output reg [DATA_WIDTH-1:0] data_out_h,
    output reg [DATA_WIDTH-1:0] data_out_v,
    output reg [DATA_WIDTH-1:0] partial_sum,
    
    input wire valid_in_h,
    input wire valid_in_v,
    output reg valid_out_h,
    output reg valid_out_v,
    
    output reg pe_active
);

reg [DATA_WIDTH-1:0] accumulator;
reg [DATA_WIDTH-1:0] product;
reg [1:0] pe_state;

localparam PE_IDLE = 2'h0;
localparam PE_COMPUTE = 2'h1;
localparam PE_FORWARD = 2'h2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        data_out_h <= '0;
        data_out_v <= '0;
        partial_sum <= '0;
        accumulator <= '0;
        product <= '0;
        valid_out_h <= 1'b0;
        valid_out_v <= 1'b0;
        pe_active <= 1'b0;
        pe_state <= PE_IDLE;
    end else if (enable) begin
        case (pe_state)
            PE_IDLE: begin
                if (valid_in_h && valid_in_v) begin
                    pe_state <= PE_COMPUTE;
                    pe_active <= 1'b1;
                end
            end
            
            PE_COMPUTE: begin
                product <= data_in_h * weight;
                accumulator <= accumulator + product;
                pe_state <= PE_FORWARD;
            end
            
            PE_FORWARD: begin
                data_out_h <= data_in_h;
                data_out_v <= data_in_v;
                partial_sum <= accumulator;
                valid_out_h <= valid_in_h;
                valid_out_v <= valid_in_v;
                pe_state <= PE_IDLE;
                pe_active <= 1'b0;
            end
        endcase
    end
end

endmodule

module accumulator_array #(
    parameter DATA_WIDTH = 32,
    parameter ARRAY_SIZE = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] partial_sums [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0],
    input wire [ARRAY_SIZE-1:0] sum_valid [ARRAY_SIZE-1:0],
    
    output reg [DATA_WIDTH-1:0] final_outputs [ARRAY_SIZE-1:0],
    output reg [ARRAY_SIZE-1:0] output_valid
);

reg [DATA_WIDTH-1:0] row_accumulators [ARRAY_SIZE-1:0];
reg [DATA_WIDTH-1:0] col_accumulators [ARRAY_SIZE-1:0];

genvar i, j;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (int k = 0; k < ARRAY_SIZE; k = k + 1) begin
            row_accumulators[k] <= '0;
            col_accumulators[k] <= '0;
            final_outputs[k] <= '0;
        end
        output_valid <= '0;
    end else if (enable) begin
        for (int m = 0; m < ARRAY_SIZE; m = m + 1) begin
            row_accumulators[m] = '0;
            col_accumulators[m] = '0;
            
            for (int n = 0; n < ARRAY_SIZE; n = n + 1) begin
                if (sum_valid[m][n]) begin
                    row_accumulators[m] = row_accumulators[m] + partial_sums[m][n];
                end
                if (sum_valid[n][m]) begin
                    col_accumulators[m] = col_accumulators[m] + partial_sums[n][m];
                end
            end
            
            final_outputs[m] <= row_accumulators[m] + col_accumulators[m];
            output_valid[m] <= |sum_valid[m];
        end
    end
end

endmodule

module weight_buffer #(
    parameter DATA_WIDTH = 32,
    parameter ARRAY_SIZE = 8,
    parameter BUFFER_DEPTH = 256
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [7:0] weight_addr,
    input wire [DATA_WIDTH-1:0] weight_data,
    input wire weight_wr_en,
    
    output reg [DATA_WIDTH-1:0] weights_out [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0],
    output reg weights_valid,
    
    input wire [7:0] config_mode,
    output wire buffer_ready
);

reg [DATA_WIDTH-1:0] weight_memory [BUFFER_DEPTH-1:0];
reg [7:0] read_addr;
reg [2:0] load_state;

localparam LOAD_IDLE = 3'h0;
localparam LOAD_WEIGHTS = 3'h1;
localparam DISTRIBUTE = 3'h2;
localparam LOAD_DONE = 3'h3;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        read_addr <= 8'h0;
        load_state <= LOAD_IDLE;
        weights_valid <= 1'b0;
        for (int i = 0; i < BUFFER_DEPTH; i = i + 1) begin
            weight_memory[i] <= '0;
        end
        for (int j = 0; j < ARRAY_SIZE; j = j + 1) begin
            for (int k = 0; k < ARRAY_SIZE; k = k + 1) begin
                weights_out[j][k] <= '0;
            end
        end
    end else if (enable) begin
        if (weight_wr_en && weight_addr < BUFFER_DEPTH) begin
            weight_memory[weight_addr] <= weight_data;
        end
        
        case (load_state)
            LOAD_IDLE: begin
                if (config_mode == 8'h01) begin
                    load_state <= LOAD_WEIGHTS;
                    read_addr <= 8'h0;
                end
            end
            
            LOAD_WEIGHTS: begin
                if (read_addr < ARRAY_SIZE * ARRAY_SIZE) begin
                    weights_out[read_addr / ARRAY_SIZE][read_addr % ARRAY_SIZE] <= 
                        weight_memory[read_addr];
                    read_addr <= read_addr + 1;
                end else begin
                    load_state <= DISTRIBUTE;
                end
            end
            
            DISTRIBUTE: begin
                weights_valid <= 1'b1;
                load_state <= LOAD_DONE;
            end
            
            LOAD_DONE: begin
                if (config_mode != 8'h01) begin
                    load_state <= LOAD_IDLE;
                    weights_valid <= 1'b0;
                end
            end
        endcase
    end
end

assign buffer_ready = (load_state == LOAD_IDLE);

endmodule

module dataflow_controller #(
    parameter DATA_WIDTH = 32,
    parameter ARRAY_SIZE = 8,
    parameter FIFO_DEPTH = 64
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] input_data,
    input wire input_valid,
    output wire input_ready,
    
    output wire [DATA_WIDTH-1:0] array_inputs [ARRAY_SIZE-1:0],
    output wire [ARRAY_SIZE-1:0] array_input_valid,
    input wire [ARRAY_SIZE-1:0] array_input_ready,
    
    input wire [DATA_WIDTH-1:0] array_outputs [ARRAY_SIZE-1:0],
    input wire [ARRAY_SIZE-1:0] array_output_valid,
    output wire [ARRAY_SIZE-1:0] array_output_ready,
    
    output wire [DATA_WIDTH-1:0] output_data,
    output wire output_valid,
    input wire output_ready,
    
    input wire [2:0] dataflow_mode
);

wire [DATA_WIDTH-1:0] input_fifo_data;
wire input_fifo_full, input_fifo_empty;
wire input_fifo_wr_en, input_fifo_rd_en;

wire [DATA_WIDTH-1:0] output_fifo_data;
wire output_fifo_full, output_fifo_empty;
wire output_fifo_wr_en, output_fifo_rd_en;

reg [7:0] distribution_counter;
reg [2:0] distribution_mode;

fifo_buffer #(
    .DATA_WIDTH(DATA_WIDTH),
    .FIFO_DEPTH(FIFO_DEPTH)
) input_fifo (
    .clk(clk),
    .rst_n(rst_n),
    .wr_data(input_data),
    .wr_en(input_fifo_wr_en),
    .rd_data(input_fifo_data),
    .rd_en(input_fifo_rd_en),
    .full(input_fifo_full),
    .empty(input_fifo_empty)
);

fifo_buffer #(
    .DATA_WIDTH(DATA_WIDTH),
    .FIFO_DEPTH(FIFO_DEPTH)
) output_fifo (
    .clk(clk),
    .rst_n(rst_n),
    .wr_data(output_fifo_data),
    .wr_en(output_fifo_wr_en),
    .rd_data(output_data),
    .rd_en(output_fifo_rd_en),
    .full(output_fifo_full),
    .empty(output_fifo_empty)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        distribution_counter <= 8'h0;
        distribution_mode <= 3'h0;
    end else if (enable) begin
        distribution_mode <= dataflow_mode;
        
        if (input_fifo_rd_en) begin
            distribution_counter <= distribution_counter + 1;
            if (distribution_counter >= ARRAY_SIZE - 1) begin
                distribution_counter <= 8'h0;
            end
        end
    end
end

assign input_ready = !input_fifo_full;
assign input_fifo_wr_en = input_valid && !input_fifo_full;
assign input_fifo_rd_en = !input_fifo_empty && |array_input_ready;

genvar i;
generate
    for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : data_distribution
        assign array_inputs[i] = (distribution_mode == 3'h0) ? input_fifo_data :
                                (distribution_mode == 3'h1) ? (distribution_counter == i ? input_fifo_data : '0) :
                                input_fifo_data;
        assign array_input_valid[i] = !input_fifo_empty && array_input_ready[i];
    end
endgenerate

assign output_fifo_data = array_outputs[0];
assign output_fifo_wr_en = array_output_valid[0] && !output_fifo_full;
assign array_output_ready = {ARRAY_SIZE{!output_fifo_full}};

assign output_valid = !output_fifo_empty;
assign output_fifo_rd_en = output_ready && !output_fifo_empty;

endmodule

module fifo_buffer #(
    parameter DATA_WIDTH = 32,
    parameter FIFO_DEPTH = 64,
    parameter ADDR_WIDTH = $clog2(FIFO_DEPTH)
)(
    input wire clk,
    input wire rst_n,
    
    input wire [DATA_WIDTH-1:0] wr_data,
    input wire wr_en,
    
    output reg [DATA_WIDTH-1:0] rd_data,
    input wire rd_en,
    
    output wire full,
    output wire empty
);

reg [DATA_WIDTH-1:0] memory [FIFO_DEPTH-1:0];
reg [ADDR_WIDTH:0] wr_ptr, rd_ptr;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_ptr <= '0;
        rd_ptr <= '0;
        rd_data <= '0;
    end else begin
        if (wr_en && !full) begin
            memory[wr_ptr[ADDR_WIDTH-1:0]] <= wr_data;
            wr_ptr <= wr_ptr + 1;
        end
        
        if (rd_en && !empty) begin
            rd_data <= memory[rd_ptr[ADDR_WIDTH-1:0]];
            rd_ptr <= rd_ptr + 1;
        end
    end
end

assign full = (wr_ptr[ADDR_WIDTH] != rd_ptr[ADDR_WIDTH]) && 
              (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]);
assign empty = (wr_ptr == rd_ptr);

endmodule 