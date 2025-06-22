module kan_multi_core #(
    parameter NUM_CORES = 8,
    parameter DATA_WIDTH = 16,
    parameter NUM_PES_PER_CORE = 32,
    parameter SPLINE_ORDER = 3,
    parameter GRID_SIZE = 64
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] input_data [0:NUM_CORES-1][0:15],
    input wire [31:0] control_register,
    input wire [7:0] core_select,
    output reg [DATA_WIDTH-1:0] output_data [0:NUM_CORES-1][0:15],
    output reg [NUM_CORES-1:0] core_ready,
    output reg computation_complete
);

    wire [DATA_WIDTH-1:0] core_outputs [0:NUM_CORES-1][0:NUM_PES_PER_CORE-1];
    wire [NUM_CORES-1:0] core_valid;
    reg [NUM_CORES-1:0] core_enable;
    reg [2:0] scheduler_state;
    reg [3:0] current_core;
    reg [7:0] load_balancer [0:NUM_CORES-1];
    
    genvar i;
    generate
        for (i = 0; i < NUM_CORES; i = i + 1) begin : core_gen
            kan_processing_core #(
                .NUM_PES(NUM_PES_PER_CORE),
                .DATA_WIDTH(DATA_WIDTH),
                .SPLINE_ORDER(SPLINE_ORDER),
                .GRID_SIZE(GRID_SIZE)
            ) kan_core_inst (
                .clk(clk),
                .rst_n(rst_n),
                .enable(enable & core_enable[i]),
                .core_inputs(input_data[i]),
                .spline_coeffs(),
                .knot_vectors(),
                .update_weights(control_register[i]),
                .core_outputs(core_outputs[i]),
                .core_outputs_valid(),
                .core_ready(core_ready[i])
            );
        end
    endgenerate
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer j = 0; j < NUM_CORES; j = j + 1) begin
                core_enable[j] <= 0;
                load_balancer[j] <= 0;
                for (integer k = 0; k < 16; k = k + 1) begin
                    output_data[j][k] <= 0;
                end
            end
            scheduler_state <= 0;
            current_core <= 0;
            computation_complete <= 0;
        end else begin
            case (scheduler_state)
                3'b000: begin
                    for (integer j = 0; j < NUM_CORES; j = j + 1) begin
                        if (core_select[j]) begin
                            core_enable[j] <= 1;
                            load_balancer[j] <= load_balancer[j] + 1;
                        end else begin
                            core_enable[j] <= 0;
                        end
                    end
                    scheduler_state <= 3'b001;
                    current_core <= 0;
                end
                
                3'b001: begin
                    if (current_core < NUM_CORES) begin
                        if (core_ready[current_core]) begin
                            for (integer k = 0; k < 16; k = k + 1) begin
                                if (k < NUM_PES_PER_CORE) begin
                                    output_data[current_core][k] <= core_outputs[current_core][k];
                                end
                            end
                            load_balancer[current_core] <= load_balancer[current_core] - 1;
                        end
                        current_core <= current_core + 1;
                    end else begin
                        scheduler_state <= 3'b010;
                    end
                end
                
                3'b010: begin
                    computation_complete <= &core_ready;
                    scheduler_state <= 3'b000;
                end
            endcase
        end
    end

endmodule 