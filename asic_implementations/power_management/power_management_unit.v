module power_management_unit #(
    parameter NUM_DOMAINS = 8,
    parameter VOLTAGE_WIDTH = 12,
    parameter CURRENT_WIDTH = 16,
    parameter TEMP_WIDTH = 10
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [VOLTAGE_WIDTH-1:0] supply_voltage,
    input wire [CURRENT_WIDTH-1:0] total_current,
    input wire [TEMP_WIDTH-1:0] chip_temperature,
    
    input wire [NUM_DOMAINS-1:0] domain_active,
    input wire [NUM_DOMAINS-1:0] domain_request,
    output reg [NUM_DOMAINS-1:0] domain_enable,
    output reg [NUM_DOMAINS-1:0] domain_clock_gate,
    
    output reg [VOLTAGE_WIDTH-1:0] target_voltage,
    output reg [3:0] frequency_scale,
    output reg thermal_throttle,
    output reg power_emergency,
    
    input wire dvfs_enable,
    input wire [15:0] power_budget,
    output reg [15:0] current_power,
    output reg power_valid
);

reg [VOLTAGE_WIDTH-1:0] voltage_levels [7:0];
reg [3:0] frequency_levels [7:0];
reg [7:0] current_state;
reg [7:0] target_state;

reg [15:0] power_history [15:0];
reg [3:0] history_ptr;
reg [19:0] power_accumulator;
reg [15:0] average_power;

reg [TEMP_WIDTH-1:0] temp_threshold_warn;
reg [TEMP_WIDTH-1:0] temp_threshold_crit;
reg [CURRENT_WIDTH-1:0] current_threshold_warn;
reg [CURRENT_WIDTH-1:0] current_threshold_crit;

wire thermal_warning;
wire thermal_critical;
wire current_warning;
wire current_critical;

reg [15:0] domain_power [NUM_DOMAINS-1:0];
reg [NUM_DOMAINS-1:0] domain_power_valid;

initial begin
    voltage_levels[0] = 12'h800;
    voltage_levels[1] = 12'h900;
    voltage_levels[2] = 12'hA00;
    voltage_levels[3] = 12'hB00;
    voltage_levels[4] = 12'hC00;
    voltage_levels[5] = 12'hD00;
    voltage_levels[6] = 12'hE00;
    voltage_levels[7] = 12'hFFF;
    
    frequency_levels[0] = 4'h2;
    frequency_levels[1] = 4'h3;
    frequency_levels[2] = 4'h4;
    frequency_levels[3] = 4'h6;
    frequency_levels[4] = 4'h8;
    frequency_levels[5] = 4'hA;
    frequency_levels[6] = 4'hC;
    frequency_levels[7] = 4'hF;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        temp_threshold_warn <= 10'h300;
        temp_threshold_crit <= 10'h380;
        current_threshold_warn <= 16'h8000;
        current_threshold_crit <= 16'hA000;
        current_state <= 8'h04;
        target_state <= 8'h04;
        history_ptr <= 4'h0;
        power_accumulator <= 20'h0;
        domain_enable <= '0;
        domain_clock_gate <= '0;
        for (int i = 0; i < 16; i = i + 1) begin
            power_history[i] <= 16'h0;
        end
        for (int j = 0; j < NUM_DOMAINS; j = j + 1) begin
            domain_power[j] <= 16'h0;
        end
        domain_power_valid <= '0;
    end else if (enable) begin
        current_power <= supply_voltage * total_current >> 8;
        power_valid <= 1'b1;
        
        power_history[history_ptr] <= current_power;
        history_ptr <= history_ptr + 1;
        
        power_accumulator = 20'h0;
        for (int k = 0; k < 16; k = k + 1) begin
            power_accumulator = power_accumulator + power_history[k];
        end
        average_power <= power_accumulator >> 4;
        
        for (int m = 0; m < NUM_DOMAINS; m = m + 1) begin
            if (domain_active[m]) begin
                domain_power[m] <= (current_power >> 3);
                domain_power_valid[m] <= 1'b1;
                domain_enable[m] <= domain_request[m];
                domain_clock_gate[m] <= !domain_request[m];
            end else begin
                domain_power[m] <= 16'h0;
                domain_power_valid[m] <= 1'b0;
                domain_enable[m] <= 1'b0;
                domain_clock_gate[m] <= 1'b1;
            end
        end
        
        if (dvfs_enable) begin
            if (average_power > power_budget && current_state > 0) begin
                target_state <= current_state - 1;
            end else if (average_power < (power_budget >> 1) && current_state < 7) begin
                target_state <= current_state + 1;
            end
            
            if (current_state != target_state) begin
                if (current_state < target_state) begin
                    current_state <= current_state + 1;
                end else begin
                    current_state <= current_state - 1;
                end
            end
        end
        
        target_voltage <= voltage_levels[current_state];
        frequency_scale <= frequency_levels[current_state];
        
        if (thermal_critical || current_critical) begin
            power_emergency <= 1'b1;
            current_state <= 8'h00;
            domain_enable <= '0;
            domain_clock_gate <= {NUM_DOMAINS{1'b1}};
        end else if (thermal_warning || current_warning) begin
            thermal_throttle <= 1'b1;
            if (current_state > 2) begin
                current_state <= current_state - 2;
            end
        end else begin
            power_emergency <= 1'b0;
            thermal_throttle <= 1'b0;
        end
    end
end

assign thermal_warning = (chip_temperature > temp_threshold_warn);
assign thermal_critical = (chip_temperature > temp_threshold_crit);
assign current_warning = (total_current > current_threshold_warn);
assign current_critical = (total_current > current_threshold_crit);

endmodule

module voltage_regulator #(
    parameter VOLTAGE_WIDTH = 12,
    parameter RESPONSE_CYCLES = 64
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [VOLTAGE_WIDTH-1:0] target_voltage,
    output reg [VOLTAGE_WIDTH-1:0] output_voltage,
    output reg voltage_ready,
    
    input wire [VOLTAGE_WIDTH-1:0] min_voltage,
    input wire [VOLTAGE_WIDTH-1:0] max_voltage,
    output wire voltage_error
);

reg [VOLTAGE_WIDTH-1:0] current_voltage;
reg [7:0] transition_counter;
reg voltage_transitioning;
wire [VOLTAGE_WIDTH-1:0] voltage_step;

assign voltage_step = (target_voltage > current_voltage) ? 12'h010 : 12'hFF0;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_voltage <= 12'h800;
        output_voltage <= 12'h800;
        transition_counter <= 8'h0;
        voltage_transitioning <= 1'b0;
        voltage_ready <= 1'b1;
    end else if (enable) begin
        if (target_voltage != current_voltage && !voltage_transitioning) begin
            voltage_transitioning <= 1'b1;
            voltage_ready <= 1'b0;
            transition_counter <= 8'h0;
        end
        
        if (voltage_transitioning) begin
            transition_counter <= transition_counter + 1;
            
            if (transition_counter >= RESPONSE_CYCLES) begin
                if (target_voltage > current_voltage) begin
                    current_voltage <= current_voltage + voltage_step;
                end else begin
                    current_voltage <= current_voltage - voltage_step;
                end
                
                if (((target_voltage > current_voltage) && 
                     (current_voltage + voltage_step >= target_voltage)) ||
                    ((target_voltage < current_voltage) && 
                     (current_voltage - voltage_step <= target_voltage))) begin
                    current_voltage <= target_voltage;
                    voltage_transitioning <= 1'b0;
                    voltage_ready <= 1'b1;
                end
                
                transition_counter <= 8'h0;
            end
        end
        
        output_voltage <= current_voltage;
    end
end

assign voltage_error = (target_voltage < min_voltage) || (target_voltage > max_voltage);

endmodule

module clock_gating_controller #(
    parameter NUM_DOMAINS = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [NUM_DOMAINS-1:0] domain_clock_gate,
    input wire [NUM_DOMAINS-1:0] domain_enable,
    input wire [3:0] frequency_scale,
    
    output reg [NUM_DOMAINS-1:0] gated_clocks,
    output reg scaled_clock,
    output wire power_savings_active
);

reg [3:0] frequency_counter;
reg clock_divider_enable;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        gated_clocks <= '0;
        frequency_counter <= 4'h0;
        clock_divider_enable <= 1'b0;
        scaled_clock <= 1'b0;
    end else if (enable) begin
        frequency_counter <= frequency_counter + 1;
        
        if (frequency_counter >= frequency_scale) begin
            frequency_counter <= 4'h0;
            clock_divider_enable <= !clock_divider_enable;
        end
        
        scaled_clock <= clock_divider_enable;
        
        for (int i = 0; i < NUM_DOMAINS; i = i + 1) begin
            if (domain_clock_gate[i]) begin
                gated_clocks[i] <= 1'b0;
            end else if (domain_enable[i]) begin
                gated_clocks[i] <= scaled_clock;
            end else begin
                gated_clocks[i] <= 1'b0;
            end
        end
    end
end

assign power_savings_active = |domain_clock_gate;

endmodule

module power_monitor #(
    parameter VOLTAGE_WIDTH = 12,
    parameter CURRENT_WIDTH = 16,
    parameter NUM_DOMAINS = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [VOLTAGE_WIDTH-1:0] supply_voltage,
    input wire [CURRENT_WIDTH-1:0] supply_current,
    input wire [NUM_DOMAINS-1:0] domain_active,
    
    output reg [15:0] total_power,
    output reg [15:0] domain_power [NUM_DOMAINS-1:0],
    output reg [31:0] energy_accumulator,
    output reg power_valid
);

reg [31:0] energy_counter;
reg [15:0] instantaneous_power;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        total_power <= 16'h0;
        energy_accumulator <= 32'h0;
        energy_counter <= 32'h0;
        power_valid <= 1'b0;
        for (int i = 0; i < NUM_DOMAINS; i = i + 1) begin
            domain_power[i] <= 16'h0;
        end
    end else if (enable) begin
        instantaneous_power = (supply_voltage * supply_current) >> 8;
        total_power <= instantaneous_power;
        power_valid <= 1'b1;
        
        energy_counter <= energy_counter + 1;
        energy_accumulator <= energy_accumulator + instantaneous_power;
        
        for (int j = 0; j < NUM_DOMAINS; j = j + 1) begin
            if (domain_active[j]) begin
                domain_power[j] <= instantaneous_power >> 3;
            end else begin
                domain_power[j] <= 16'h0;
            end
        end
    end
end

endmodule 