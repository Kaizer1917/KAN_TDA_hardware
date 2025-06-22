module tda_acceleration_unit #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 12,
    parameter NUM_REDUCTION_UNITS = 8,
    parameter MAX_SIMPLICES = 4096
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] simplex_data,
    input wire [ADDR_WIDTH-1:0] simplex_addr,
    input wire simplex_valid,
    input wire compute_persistence,
    output reg [DATA_WIDTH-1:0] persistence_pairs [0:MAX_SIMPLICES-1][1:0],
    output reg [ADDR_WIDTH-1:0] num_pairs,
    output reg computation_complete
);

    wire [DATA_WIDTH-1:0] sbmp_values;
    wire [ADDR_WIDTH-1:0] sbmp_row_indices;
    wire sbmp_data_valid;
    wire [DATA_WIDTH-1:0] pre_reduced_columns [0:NUM_REDUCTION_UNITS-1];
    wire [ADDR_WIDTH-1:0] pre_pivot_indices [0:NUM_REDUCTION_UNITS-1];
    wire pre_reduction_complete;
    
    reg [DATA_WIDTH-1:0] column_data_in [0:NUM_REDUCTION_UNITS-1];
    reg [ADDR_WIDTH-1:0] column_addr_in [0:NUM_REDUCTION_UNITS-1];
    reg column_valid_in [0:NUM_REDUCTION_UNITS-1];
    reg [3:0] tda_state;
    reg [ADDR_WIDTH-1:0] current_column;
    reg [7:0] unit_index;
    reg [ADDR_WIDTH-1:0] pair_index;

    sparse_boundary_matrix_processor #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) sbmp_inst (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .values_in(simplex_data),
        .row_indices_in(simplex_addr),
        .col_ptr_in(current_column),
        .read_addr(current_column),
        .write_enable(simplex_valid),
        .values_out(sbmp_values),
        .row_indices_out(sbmp_row_indices),
        .data_valid(sbmp_data_valid)
    );

    parallel_reduction_engine #(
        .NUM_REDUCTION_UNITS(NUM_REDUCTION_UNITS),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) pre_inst (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .column_data(column_data_in),
        .column_addr(column_addr_in),
        .column_valid(column_valid_in),
        .reduced_columns(pre_reduced_columns),
        .pivot_indices(pre_pivot_indices),
        .reduction_complete(pre_reduction_complete)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            computation_complete <= 0;
            num_pairs <= 0;
            tda_state <= 0;
            current_column <= 0;
            unit_index <= 0;
            pair_index <= 0;
            for (integer i = 0; i < NUM_REDUCTION_UNITS; i = i + 1) begin
                column_data_in[i] <= 0;
                column_addr_in[i] <= 0;
                column_valid_in[i] <= 0;
            end
        end else if (compute_persistence) begin
            case (tda_state)
                4'b0000: begin
                    current_column <= 0;
                    unit_index <= 0;
                    pair_index <= 0;
                    tda_state <= 4'b0001;
                    computation_complete <= 0;
                end
                4'b0001: begin
                    if (sbmp_data_valid) begin
                        column_data_in[unit_index] <= sbmp_values;
                        column_addr_in[unit_index] <= sbmp_row_indices;
                        column_valid_in[unit_index] <= 1;
                        unit_index <= unit_index + 1;
                        if (unit_index >= NUM_REDUCTION_UNITS - 1) begin
                            tda_state <= 4'b0010;
                            unit_index <= 0;
                        end
                    end
                end
                4'b0010: begin
                    for (integer i = 0; i < NUM_REDUCTION_UNITS; i = i + 1) begin
                        column_valid_in[i] <= 0;
                    end
                    if (pre_reduction_complete) begin
                        tda_state <= 4'b0011;
                    end
                end
                4'b0011: begin
                    for (integer i = 0; i < NUM_REDUCTION_UNITS; i = i + 1) begin
                        if (pre_pivot_indices[i] != 0) begin
                            persistence_pairs[pair_index][0] <= current_column + i;
                            persistence_pairs[pair_index][1] <= pre_pivot_indices[i];
                            pair_index <= pair_index + 1;
                        end
                    end
                    current_column <= current_column + NUM_REDUCTION_UNITS;
                    if (current_column >= MAX_SIMPLICES - NUM_REDUCTION_UNITS) begin
                        tda_state <= 4'b0100;
                    end else begin
                        tda_state <= 4'b0001;
                        unit_index <= 0;
                    end
                end
                4'b0100: begin
                    num_pairs <= pair_index;
                    computation_complete <= 1;
                    tda_state <= 4'b0000;
                end
            endcase
        end
    end

endmodule 