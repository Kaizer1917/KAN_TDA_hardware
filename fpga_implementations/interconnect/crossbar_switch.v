module crossbar_switch #(
    parameter DATA_WIDTH = 256,
    parameter NUM_INPUTS = 8,
    parameter NUM_OUTPUTS = 8,
    parameter ADDR_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire [NUM_INPUTS-1:0] valid_in,
    input wire [DATA_WIDTH-1:0] data_in [0:NUM_INPUTS-1],
    input wire [ADDR_WIDTH-1:0] dest_addr_in [0:NUM_INPUTS-1],
    input wire [3:0] priority_in [0:NUM_INPUTS-1],
    output reg [NUM_INPUTS-1:0] ready_out,
    output reg [NUM_OUTPUTS-1:0] valid_out,
    output reg [DATA_WIDTH-1:0] data_out [0:NUM_OUTPUTS-1],
    output reg [ADDR_WIDTH-1:0] dest_addr_out [0:NUM_OUTPUTS-1],
    input wire [NUM_OUTPUTS-1:0] ready_in
);

    reg [NUM_INPUTS-1:0] request_matrix [0:NUM_OUTPUTS-1];
    reg [NUM_INPUTS-1:0] grant_matrix [0:NUM_OUTPUTS-1];
    reg [3:0] input_priority [0:NUM_INPUTS-1];
    reg [2:0] output_select [0:NUM_OUTPUTS-1];
    reg [1:0] arbitration_state;
    reg [3:0] current_output;
    
    function [2:0] decode_destination;
        input [ADDR_WIDTH-1:0] addr;
        begin
            decode_destination = addr[2:0];
        end
    endfunction
    
    function [2:0] priority_select;
        input [NUM_INPUTS-1:0] requests;
        input [3:0] priorities [0:NUM_INPUTS-1];
        reg [3:0] max_priority;
        reg [2:0] selected_input;
        integer k;
        begin
            max_priority = 0;
            selected_input = 0;
            for (k = 0; k < NUM_INPUTS; k = k + 1) begin
                if (requests[k] && priorities[k] > max_priority) begin
                    max_priority = priorities[k];
                    selected_input = k;
                end
            end
            priority_select = selected_input;
        end
    endfunction
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < NUM_OUTPUTS; i = i + 1) begin
                for (integer j = 0; j < NUM_INPUTS; j = j + 1) begin
                    request_matrix[i][j] <= 0;
                    grant_matrix[i][j] <= 0;
                end
                valid_out[i] <= 0;
                output_select[i] <= 0;
            end
            for (integer i = 0; i < NUM_INPUTS; i = i + 1) begin
                ready_out[i] <= 1;
                input_priority[i] <= priority_in[i];
            end
            arbitration_state <= 0;
            current_output <= 0;
        end else begin
            case (arbitration_state)
                2'b00: begin
                    for (integer i = 0; i < NUM_INPUTS; i = i + 1) begin
                        if (valid_in[i]) begin
                            request_matrix[decode_destination(dest_addr_in[i])][i] <= 1;
                            input_priority[i] <= priority_in[i];
                        end else begin
                            for (integer j = 0; j < NUM_OUTPUTS; j = j + 1) begin
                                request_matrix[j][i] <= 0;
                            end
                        end
                    end
                    arbitration_state <= 2'b01;
                    current_output <= 0;
                end
                
                2'b01: begin
                    if (current_output < NUM_OUTPUTS) begin
                        if (|request_matrix[current_output]) begin
                            output_select[current_output] <= priority_select(request_matrix[current_output], input_priority);
                            grant_matrix[current_output][priority_select(request_matrix[current_output], input_priority)] <= 1;
                        end
                        current_output <= current_output + 1;
                    end else begin
                        arbitration_state <= 2'b10;
                    end
                end
                
                2'b10: begin
                    for (integer i = 0; i < NUM_OUTPUTS; i = i + 1) begin
                        if (|grant_matrix[i] && ready_in[i]) begin
                            data_out[i] <= data_in[output_select[i]];
                            dest_addr_out[i] <= dest_addr_in[output_select[i]];
                            valid_out[i] <= 1;
                            ready_out[output_select[i]] <= 0;
                        end else begin
                            valid_out[i] <= 0;
                        end
                    end
                    arbitration_state <= 2'b11;
                end
                
                2'b11: begin
                    for (integer i = 0; i < NUM_OUTPUTS; i = i + 1) begin
                        for (integer j = 0; j < NUM_INPUTS; j = j + 1) begin
                            if (grant_matrix[i][j] && ready_in[i]) begin
                                grant_matrix[i][j] <= 0;
                                request_matrix[i][j] <= 0;
                                ready_out[j] <= 1;
                            end
                        end
                    end
                    arbitration_state <= 2'b00;
                end
            endcase
        end
    end

endmodule 