module parallel_reduction_engine #(
    parameter NUM_REDUCTION_UNITS = 8,
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 12,
    parameter COLUMN_BUFFER_SIZE = 256
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] column_data [0:NUM_REDUCTION_UNITS-1],
    input wire [ADDR_WIDTH-1:0] column_addr [0:NUM_REDUCTION_UNITS-1],
    input wire column_valid [0:NUM_REDUCTION_UNITS-1],
    output reg [DATA_WIDTH-1:0] reduced_columns [0:NUM_REDUCTION_UNITS-1],
    output reg [ADDR_WIDTH-1:0] pivot_indices [0:NUM_REDUCTION_UNITS-1],
    output reg reduction_complete
);

    reg [DATA_WIDTH-1:0] column_buffers [0:NUM_REDUCTION_UNITS-1][0:COLUMN_BUFFER_SIZE-1];
    reg [ADDR_WIDTH-1:0] pivot_tracker [0:NUM_REDUCTION_UNITS-1];
    reg [DATA_WIDTH-1:0] xor_results [0:NUM_REDUCTION_UNITS-1];
    reg [3:0] reduction_state [0:NUM_REDUCTION_UNITS-1];
    reg [7:0] buffer_index [0:NUM_REDUCTION_UNITS-1];
    reg [ADDR_WIDTH-1:0] lowest_one [0:NUM_REDUCTION_UNITS-1];
    wire [NUM_REDUCTION_UNITS-1:0] unit_complete;

    genvar i;
    generate
        for (i = 0; i < NUM_REDUCTION_UNITS; i = i + 1) begin : reduction_unit_gen
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    reduced_columns[i] <= 0;
                    pivot_indices[i] <= 0;
                    pivot_tracker[i] <= 0;
                    reduction_state[i] <= 0;
                    buffer_index[i] <= 0;
                    lowest_one[i] <= 0;
                end else if (enable && column_valid[i]) begin
                    case (reduction_state[i])
                        4'b0000: begin
                            column_buffers[i][buffer_index[i]] <= column_data[i];
                            buffer_index[i] <= buffer_index[i] + 1;
                            reduction_state[i] <= 4'b0001;
                        end
                        4'b0001: begin
                            lowest_one[i] <= column_addr[i];
                            reduction_state[i] <= 4'b0010;
                        end
                        4'b0010: begin
                            if (pivot_tracker[i] != 0) begin
                                xor_results[i] <= column_buffers[i][buffer_index[i]] ^ 
                                                column_buffers[pivot_tracker[i]][buffer_index[i]];
                                reduction_state[i] <= 4'b0011;
                            end else begin
                                reduction_state[i] <= 4'b0100;
                            end
                        end
                        4'b0011: begin
                            column_buffers[i][buffer_index[i]] <= xor_results[i];
                            if (buffer_index[i] > 0) begin
                                buffer_index[i] <= buffer_index[i] - 1;
                                reduction_state[i] <= 4'b0010;
                            end else begin
                                reduction_state[i] <= 4'b0100;
                            end
                        end
                        4'b0100: begin
                            pivot_indices[i] <= lowest_one[i];
                            reduced_columns[i] <= column_buffers[i][0];
                            reduction_state[i] <= 4'b0101;
                        end
                        4'b0101: begin
                            reduction_state[i] <= 4'b0000;
                            buffer_index[i] <= 0;
                        end
                    endcase
                end
            end
            
            assign unit_complete[i] = (reduction_state[i] == 4'b0101);
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reduction_complete <= 0;
        end else begin
            reduction_complete <= &unit_complete;
        end
    end

endmodule 