module kan_systolic_array #(
    parameter ARRAY_SIZE = 8,
    parameter DATA_WIDTH = 16,
    parameter COEFF_WIDTH = 16,
    parameter GRID_SIZE = 8,
    parameter NUM_INPUTS = 4
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] input_data [0:ARRAY_SIZE-1][0:NUM_INPUTS-1],
    input wire [8:0] coeff_addr [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    input wire coeff_we [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    input wire [COEFF_WIDTH-1:0] coeff_data [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output wire [DATA_WIDTH-1:0] output_data [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output wire output_valid [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    wire [DATA_WIDTH-1:0] horizontal_data [0:ARRAY_SIZE-1][0:ARRAY_SIZE][0:NUM_INPUTS-1];
    wire [DATA_WIDTH-1:0] vertical_data [0:ARRAY_SIZE][0:ARRAY_SIZE-1][0:NUM_INPUTS-1];
    wire horizontal_valid [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    wire vertical_valid [0:ARRAY_SIZE][0:ARRAY_SIZE-1];

    genvar i, j, k;
    
    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : row_gen
            for (k = 0; k < NUM_INPUTS; k = k + 1) begin : input_assign
                assign horizontal_data[i][0][k] = input_data[i][k];
            end
            assign horizontal_valid[i][0] = enable;
        end
    endgenerate

    generate
        for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : col_gen
            for (k = 0; k < NUM_INPUTS; k = k + 1) begin : input_assign_v
                assign vertical_data[0][j][k] = input_data[0][k];
            end
            assign vertical_valid[0][j] = enable;
        end
    endgenerate

    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : pe_row_gen
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : pe_col_gen
                kan_processing_element #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .COEFF_WIDTH(COEFF_WIDTH),
                    .GRID_SIZE(GRID_SIZE),
                    .NUM_INPUTS(NUM_INPUTS)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(horizontal_valid[i][j] && vertical_valid[i][j]),
                    .inputs(horizontal_data[i][j]),
                    .coeff_addr(coeff_addr[i][j]),
                    .coeff_we(coeff_we[i][j]),
                    .coeff_data_in(coeff_data[i][j]),
                    .pe_output(output_data[i][j]),
                    .output_valid(output_valid[i][j])
                );
                
                if (j < ARRAY_SIZE - 1) begin : horizontal_connect
                    for (k = 0; k < NUM_INPUTS; k = k + 1) begin : h_data_connect
                        assign horizontal_data[i][j+1][k] = horizontal_data[i][j][k];
                    end
                    assign horizontal_valid[i][j+1] = horizontal_valid[i][j];
                end
                
                if (i < ARRAY_SIZE - 1) begin : vertical_connect
                    for (k = 0; k < NUM_INPUTS; k = k + 1) begin : v_data_connect
                        assign vertical_data[i+1][j][k] = vertical_data[i][j][k];
                    end
                    assign vertical_valid[i+1][j] = vertical_valid[i][j];
                end
            end
        end
    endgenerate

endmodule 