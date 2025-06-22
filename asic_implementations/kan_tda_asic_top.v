module kan_tda_asic_top #(
    parameter NUM_KAN_CORES = 16,
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 12,
    parameter NUM_TDA_UNITS = 4,
    parameter L3_CACHE_SIZE = 2097152
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] input_data [0:255],
    input wire [31:0] control_register,
    input wire [ADDR_WIDTH-1:0] memory_addr,
    input wire memory_we,
    input wire [DATA_WIDTH-1:0] memory_data_in,
    output wire [DATA_WIDTH-1:0] memory_data_out,
    output wire [DATA_WIDTH-1:0] output_data [0:255],
    output wire computation_complete,
    output wire [15:0] power_status
);

    wire [DATA_WIDTH-1:0] kan_core_outputs [0:NUM_KAN_CORES-1][0:63];
    wire kan_core_ready [0:NUM_KAN_CORES-1];
    wire [DATA_WIDTH-1:0] tda_persistence_pairs [0:NUM_TDA_UNITS-1][0:4095][1:0];
    wire tda_computation_complete [0:NUM_TDA_UNITS-1];
    
    wire [NUM_KAN_CORES-1:0] kan_domain_active;
    wire [NUM_TDA_UNITS-1:0] tda_domain_active;
    wire [NUM_KAN_CORES + NUM_TDA_UNITS - 1:0] clock_gates;
    wire [NUM_KAN_CORES + NUM_TDA_UNITS - 1:0] power_gates;
    
    reg [DATA_WIDTH-1:0] l3_cache [0:L3_CACHE_SIZE-1];
    reg [DATA_WIDTH-1:0] l3_cache_out;
    reg [7:0] workload_metric;
    reg [7:0] temperature_reading;
    reg [3:0] system_state;
    reg [7:0] core_index;
    
    wire noc_clk = clk & ~clock_gates[0];
    wire system_ready = &kan_core_ready & &tda_computation_complete;

    genvar i;
    generate
        for (i = 0; i < NUM_KAN_CORES; i = i + 1) begin : kan_core_gen
            kan_processing_core #(
                .NUM_PES(64),
                .DATA_WIDTH(DATA_WIDTH)
            ) kan_core_inst (
                .clk(clk & ~clock_gates[i]),
                .rst_n(rst_n & ~power_gates[i]),
                .enable(enable & kan_domain_active[i]),
                .core_inputs(input_data[i*16 +: 16]),
                .l2_cache_addr(memory_addr),
                .l2_cache_we(memory_we && (memory_addr[23:16] == i)),
                .l2_cache_data_in(memory_data_in),
                .l2_cache_data_out(),
                .core_outputs(kan_core_outputs[i]),
                .core_outputs_valid(),
                .core_ready(kan_core_ready[i])
            );
            
            assign kan_domain_active[i] = control_register[i];
        end
    endgenerate

    generate
        for (i = 0; i < NUM_TDA_UNITS; i = i + 1) begin : tda_unit_gen
            tda_acceleration_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .ADDR_WIDTH(ADDR_WIDTH)
            ) tda_unit_inst (
                .clk(clk & ~clock_gates[NUM_KAN_CORES + i]),
                .rst_n(rst_n & ~power_gates[NUM_KAN_CORES + i]),
                .enable(enable & tda_domain_active[i]),
                .simplex_data(input_data[i]),
                .simplex_addr(memory_addr),
                .simplex_valid(memory_we),
                .compute_persistence(control_register[16 + i]),
                .persistence_pairs(tda_persistence_pairs[i]),
                .num_pairs(),
                .computation_complete(tda_computation_complete[i])
            );
            
            assign tda_domain_active[i] = control_register[16 + i];
        end
    endgenerate

    power_management_unit #(
        .NUM_POWER_DOMAINS(NUM_KAN_CORES + NUM_TDA_UNITS)
    ) pmu_inst (
        .clk(clk),
        .rst_n(rst_n),
        .workload_indicator(workload_metric),
        .temperature_sensor(temperature_reading),
        .domain_active({tda_domain_active, kan_domain_active}),
        .clock_gates(clock_gates),
        .power_gates(power_gates),
        .voltage_level(),
        .frequency_scale(),
        .power_consumption_estimate(power_status)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            system_state <= 0;
            core_index <= 0;
            workload_metric <= 0;
            temperature_reading <= 8'd150;
            l3_cache_out <= 0;
        end else begin
            if (memory_we && memory_addr < L3_CACHE_SIZE) begin
                l3_cache[memory_addr] <= memory_data_in;
            end else if (memory_addr < L3_CACHE_SIZE) begin
                l3_cache_out <= l3_cache[memory_addr];
            end
            
            case (system_state)
                4'b0000: begin
                    workload_metric <= {kan_domain_active, tda_domain_active};
                    system_state <= 4'b0001;
                end
                4'b0001: begin
                    if (core_index < NUM_KAN_CORES) begin
                        output_data[core_index*4 +: 4] <= kan_core_outputs[core_index][0:3];
                        core_index <= core_index + 1;
                    end else begin
                        system_state <= 4'b0010;
                        core_index <= 0;
                    end
                end
                4'b0010: begin
                    system_state <= 4'b0000;
                end
            endcase
            
            temperature_reading <= temperature_reading + 
                                 (workload_metric > 8'd128 ? 8'd1 : 
                                  (workload_metric < 8'd64 ? -8'd1 : 8'd0));
        end
    end

    assign memory_data_out = l3_cache_out;
    assign computation_complete = system_ready;

endmodule 