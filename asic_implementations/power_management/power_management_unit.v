module power_management_unit #(
    parameter NUM_POWER_DOMAINS = 16,
    parameter VOLTAGE_LEVELS = 8,
    parameter FREQ_LEVELS = 16
)(
    input wire clk,
    input wire rst_n,
    input wire [7:0] workload_indicator,
    input wire [7:0] temperature_sensor,
    input wire [NUM_POWER_DOMAINS-1:0] domain_active,
    output reg [NUM_POWER_DOMAINS-1:0] clock_gates,
    output reg [NUM_POWER_DOMAINS-1:0] power_gates,
    output reg [2:0] voltage_level [0:NUM_POWER_DOMAINS-1],
    output reg [3:0] frequency_scale [0:NUM_POWER_DOMAINS-1],
    output reg [15:0] power_consumption_estimate
);

    reg [7:0] activity_counters [0:NUM_POWER_DOMAINS-1];
    reg [15:0] power_budget;
    reg [7:0] thermal_threshold;
    reg [3:0] dvfs_state;
    reg [4:0] domain_index;
    reg [15:0] total_power;
    
    wire [7:0] base_power_per_domain = 8'd6;
    wire [7:0] dynamic_power_factor [0:NUM_POWER_DOMAINS-1];
    wire thermal_emergency = (temperature_sensor > thermal_threshold);

    genvar i;
    generate
        for (i = 0; i < NUM_POWER_DOMAINS; i = i + 1) begin : power_calc_gen
            assign dynamic_power_factor[i] = (activity_counters[i] * voltage_level[i] * frequency_scale[i]) >> 4;
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            power_budget <= 16'd100;
            thermal_threshold <= 8'd200;
            dvfs_state <= 0;
            domain_index <= 0;
            total_power <= 0;
            power_consumption_estimate <= 0;
            
            for (integer j = 0; j < NUM_POWER_DOMAINS; j = j + 1) begin
                clock_gates[j] <= 0;
                power_gates[j] <= 0;
                voltage_level[j] <= 3'd4;
                frequency_scale[j] <= 4'd8;
                activity_counters[j] <= 0;
            end
        end else begin
            case (dvfs_state)
                4'b0000: begin
                    for (integer j = 0; j < NUM_POWER_DOMAINS; j = j + 1) begin
                        activity_counters[j] <= domain_active[j] ? activity_counters[j] + 1 : 
                                              (activity_counters[j] > 0 ? activity_counters[j] - 1 : 0);
                    end
                    dvfs_state <= 4'b0001;
                    domain_index <= 0;
                end
                4'b0001: begin
                    if (domain_index < NUM_POWER_DOMAINS) begin
                        if (!domain_active[domain_index]) begin
                            clock_gates[domain_index] <= 1;
                            if (activity_counters[domain_index] == 0) begin
                                power_gates[domain_index] <= 1;
                                voltage_level[domain_index] <= 0;
                            end
                        end else begin
                            clock_gates[domain_index] <= 0;
                            power_gates[domain_index] <= 0;
                            
                            if (thermal_emergency) begin
                                voltage_level[domain_index] <= (voltage_level[domain_index] > 1) ? 
                                                             voltage_level[domain_index] - 1 : 1;
                                frequency_scale[domain_index] <= (frequency_scale[domain_index] > 2) ? 
                                                                frequency_scale[domain_index] - 1 : 2;
                            end else if (workload_indicator > 8'd200) begin
                                voltage_level[domain_index] <= (voltage_level[domain_index] < 7) ? 
                                                             voltage_level[domain_index] + 1 : 7;
                                frequency_scale[domain_index] <= (frequency_scale[domain_index] < 15) ? 
                                                                frequency_scale[domain_index] + 1 : 15;
                            end else if (workload_indicator < 8'd50) begin
                                voltage_level[domain_index] <= (voltage_level[domain_index] > 2) ? 
                                                             voltage_level[domain_index] - 1 : 2;
                                frequency_scale[domain_index] <= (frequency_scale[domain_index] > 4) ? 
                                                                frequency_scale[domain_index] - 1 : 4;
                            end
                        end
                        domain_index <= domain_index + 1;
                    end else begin
                        dvfs_state <= 4'b0010;
                    end
                end
                4'b0010: begin
                    total_power <= 0;
                    for (integer j = 0; j < NUM_POWER_DOMAINS; j = j + 1) begin
                        total_power <= total_power + base_power_per_domain + dynamic_power_factor[j];
                    end
                    dvfs_state <= 4'b0011;
                end
                4'b0011: begin
                    power_consumption_estimate <= total_power;
                    dvfs_state <= 4'b0000;
                end
            endcase
        end
    end

endmodule 