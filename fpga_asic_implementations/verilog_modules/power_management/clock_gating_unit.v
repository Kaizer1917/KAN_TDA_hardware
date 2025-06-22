module clock_gating_unit #(
    parameter NUM_DOMAINS = 16
)(
    input wire ref_clk,
    input wire rst_n,
    input wire [NUM_DOMAINS-1:0] domain_enable,
    input wire [NUM_DOMAINS-1:0] activity_detect,
    input wire [7:0] power_mode,
    output reg [NUM_DOMAINS-1:0] gated_clk,
    output reg [NUM_DOMAINS-1:0] clock_valid,
    output reg [15:0] power_savings_estimate
);

    reg [NUM_DOMAINS-1:0] gate_control;
    reg [NUM_DOMAINS-1:0] activity_history [0:7];
    reg [2:0] history_index;
    reg [7:0] activity_counter [0:NUM_DOMAINS-1];
    reg [15:0] idle_cycles [0:NUM_DOMAINS-1];
    reg [3:0] power_state;
    
    wire [NUM_DOMAINS-1:0] activity_sum;
    
    genvar i;
    generate
        for (i = 0; i < NUM_DOMAINS; i = i + 1) begin : domain_gen
            assign activity_sum[i] = activity_history[0][i] + activity_history[1][i] + 
                                   activity_history[2][i] + activity_history[3][i] +
                                   activity_history[4][i] + activity_history[5][i] +
                                   activity_history[6][i] + activity_history[7][i];
            
            always @(posedge ref_clk or negedge rst_n) begin
                if (!rst_n) begin
                    gated_clk[i] <= 0;
                    clock_valid[i] <= 0;
                end else begin
                    if (gate_control[i] && domain_enable[i]) begin
                        gated_clk[i] <= ref_clk;
                        clock_valid[i] <= 1;
                    end else begin
                        gated_clk[i] <= 0;
                        clock_valid[i] <= 0;
                    end
                end
            end
        end
    endgenerate
    
    always @(posedge ref_clk or negedge rst_n) begin
        if (!rst_n) begin
            gate_control <= {NUM_DOMAINS{1'b1}};
            history_index <= 0;
            power_state <= 0;
            power_savings_estimate <= 0;
            for (integer j = 0; j < NUM_DOMAINS; j = j + 1) begin
                activity_counter[j] <= 0;
                idle_cycles[j] <= 0;
            end
            for (integer k = 0; k < 8; k = k + 1) begin
                activity_history[k] <= 0;
            end
        end else begin
            activity_history[history_index] <= activity_detect;
            history_index <= (history_index + 1) % 8;
            
            for (integer j = 0; j < NUM_DOMAINS; j = j + 1) begin
                if (activity_detect[j]) begin
                    activity_counter[j] <= activity_counter[j] + 1;
                    idle_cycles[j] <= 0;
                end else begin
                    idle_cycles[j] <= idle_cycles[j] + 1;
                end
                
                case (power_mode)
                    8'h00: begin
                        gate_control[j] <= domain_enable[j];
                    end
                    8'h01: begin
                        gate_control[j] <= domain_enable[j] && (activity_sum[j] > 2);
                    end
                    8'h02: begin
                        gate_control[j] <= domain_enable[j] && (activity_sum[j] > 4);
                    end
                    8'h03: begin
                        gate_control[j] <= domain_enable[j] && (idle_cycles[j] < 16);
                    end
                    default: begin
                        gate_control[j] <= domain_enable[j];
                    end
                endcase
            end
            
            case (power_state)
                4'b0000: begin
                    power_savings_estimate <= 0;
                    for (integer j = 0; j < NUM_DOMAINS; j = j + 1) begin
                        if (!gate_control[j] && domain_enable[j]) begin
                            power_savings_estimate <= power_savings_estimate + 100;
                        end
                    end
                    power_state <= 4'b0001;
                end
                4'b0001: begin
                    power_state <= 4'b0000;
                end
            endcase
        end
    end

endmodule 