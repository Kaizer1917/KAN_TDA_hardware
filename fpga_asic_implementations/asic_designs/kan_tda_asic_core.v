module kan_tda_asic_core #(
    parameter NUM_KAN_CORES = 16,
    parameter NUM_TDA_ENGINES = 4,
    parameter DATA_WIDTH = 16,
    parameter CACHE_SIZE = 262144,
    parameter NUM_POWER_DOMAINS = 24
)(
    input wire sys_clk,
    input wire por_rst_n,
    input wire test_mode,
    input wire scan_enable,
    input wire [31:0] config_data,
    input wire config_valid,
    
    input wire [DATA_WIDTH-1:0] data_in [0:255],
    input wire [31:0] control_reg,
    input wire data_valid,
    output reg [DATA_WIDTH-1:0] data_out [0:255],
    output reg result_valid,
    output reg computation_done,
    
    output wire [15:0] power_status,
    output wire [7:0] thermal_status,
    output wire chip_ready,
    
    input wire jtag_tck,
    input wire jtag_tdi,
    input wire jtag_tms,
    input wire jtag_trst_n,
    output wire jtag_tdo
);

    wire pll_clk, pll_locked;
    wire [NUM_POWER_DOMAINS-1:0] domain_clocks;
    wire [NUM_POWER_DOMAINS-1:0] domain_resets;
    wire [NUM_POWER_DOMAINS-1:0] power_gates;
    wire [NUM_POWER_DOMAINS-1:0] clock_gates;
    
    wire [DATA_WIDTH-1:0] kan_outputs [0:NUM_KAN_CORES-1][0:63];
    wire [NUM_KAN_CORES-1:0] kan_ready;
    wire [DATA_WIDTH-1:0] tda_outputs [0:NUM_TDA_ENGINES-1][0:255];
    wire [NUM_TDA_ENGINES-1:0] tda_ready;
    
    wire [31:0] cache_addr;
    wire [255:0] cache_data_in, cache_data_out;
    wire cache_we, cache_re, cache_ready, cache_hit;
    
    wire [7:0] workload_metric;
    wire [7:0] temperature_sensor;
    wire [15:0] voltage_monitor;
    wire [15:0] current_monitor;
    
    pll_unit pll_inst (
        .ref_clk(sys_clk),
        .rst_n(por_rst_n),
        .pll_clk(pll_clk),
        .locked(pll_locked)
    );
    
    power_management_unit #(
        .NUM_POWER_DOMAINS(NUM_POWER_DOMAINS)
    ) pmu_inst (
        .clk(pll_clk),
        .rst_n(por_rst_n & pll_locked),
        .workload_indicator(workload_metric),
        .temperature_sensor(temperature_sensor),
        .voltage_monitor(voltage_monitor),
        .current_monitor(current_monitor),
        .domain_active(control_reg[NUM_POWER_DOMAINS-1:0]),
        .clock_gates(clock_gates),
        .power_gates(power_gates),
        .domain_clocks(domain_clocks),
        .domain_resets(domain_resets),
        .power_consumption_estimate(power_status)
    );
    
    clock_gating_unit #(
        .NUM_DOMAINS(NUM_POWER_DOMAINS)
    ) cgu_inst (
        .ref_clk(pll_clk),
        .rst_n(por_rst_n & pll_locked),
        .domain_enable(control_reg[NUM_POWER_DOMAINS-1:0]),
        .activity_detect({kan_ready, tda_ready, 8'b0}),
        .power_mode(config_data[7:0]),
        .gated_clk(domain_clocks),
        .clock_valid(),
        .power_savings_estimate()
    );
    
    cache_controller #(
        .DATA_WIDTH(256),
        .ADDR_WIDTH(32),
        .CACHE_SIZE(CACHE_SIZE),
        .ASSOCIATIVITY(8),
        .BLOCK_SIZE(64)
    ) l3_cache_inst (
        .clk(domain_clocks[0]),
        .rst_n(domain_resets[0]),
        .cpu_addr(cache_addr),
        .cpu_data_in(cache_data_in),
        .cpu_we(cache_we),
        .cpu_re(cache_re),
        .cpu_data_out(cache_data_out),
        .cpu_ready(cache_ready),
        .cache_hit(cache_hit),
        .mem_addr(),
        .mem_data_out(),
        .mem_data_in(256'b0),
        .mem_we(),
        .mem_re(),
        .mem_ready(1'b1)
    );
    
    genvar i;
    generate
        for (i = 0; i < NUM_KAN_CORES; i = i + 1) begin : kan_core_gen
            kan_multi_core #(
                .NUM_CORES(1),
                .DATA_WIDTH(DATA_WIDTH),
                .NUM_PES_PER_CORE(64)
            ) kan_core_inst (
                .clk(domain_clocks[i+1]),
                .rst_n(domain_resets[i+1]),
                .enable(control_reg[i] & ~power_gates[i+1]),
                .input_data({data_in[i*16 +: 16]}),
                .control_register(config_data),
                .core_select(8'h01),
                .output_data(kan_outputs[i]),
                .core_ready(kan_ready[i]),
                .computation_complete()
            );
        end
    endgenerate
    
    generate
        for (i = 0; i < NUM_TDA_ENGINES; i = i + 1) begin : tda_engine_gen
            homology_engine #(
                .DATA_WIDTH(DATA_WIDTH),
                .MAX_SIMPLICES(2048),
                .MAX_DIMENSION(3)
            ) tda_engine_inst (
                .clk(domain_clocks[NUM_KAN_CORES+1+i]),
                .rst_n(domain_resets[NUM_KAN_CORES+1+i]),
                .enable(control_reg[NUM_KAN_CORES+i] & ~power_gates[NUM_KAN_CORES+1+i]),
                .simplex_data(data_in[i*64 +: 64]),
                .num_simplices(config_data[27:16]),
                .max_dimension(config_data[29:28]),
                .compute_start(control_reg[24+i]),
                .betti_numbers(),
                .persistence_pairs(tda_outputs[i]),
                .num_pairs(),
                .computation_complete(tda_ready[i])
            );
        end
    endgenerate
    
    jtag_tap jtag_inst (
        .tck(jtag_tck),
        .tdi(jtag_tdi),
        .tms(jtag_tms),
        .trst_n(jtag_trst_n),
        .tdo(jtag_tdo),
        .test_mode(test_mode),
        .scan_enable(scan_enable)
    );
    
    thermal_sensor thermal_inst (
        .clk(domain_clocks[0]),
        .rst_n(domain_resets[0]),
        .enable(1'b1),
        .temperature_reading(temperature_sensor),
        .thermal_status(thermal_status)
    );
    
    always @(posedge domain_clocks[0] or negedge domain_resets[0]) begin
        if (!domain_resets[0]) begin
            for (integer j = 0; j < 256; j = j + 1) begin
                data_out[j] <= 0;
            end
            result_valid <= 0;
            computation_done <= 0;
            workload_metric <= 0;
        end else begin
            workload_metric <= {kan_ready, tda_ready};
            
            if (&kan_ready && &tda_ready) begin
                for (integer j = 0; j < NUM_KAN_CORES; j = j + 1) begin
                    if (j*4 + 3 < 256) begin
                        data_out[j*4 +: 4] <= kan_outputs[j][0:3];
                    end
                end
                
                for (integer j = 0; j < NUM_TDA_ENGINES; j = j + 1) begin
                    if ((NUM_KAN_CORES*4 + j*16 + 15) < 256) begin
                        data_out[NUM_KAN_CORES*4 + j*16 +: 16] <= tda_outputs[j][0:15];
                    end
                end
                
                result_valid <= 1;
                computation_done <= 1;
            end else begin
                result_valid <= 0;
                computation_done <= 0;
            end
        end
    end
    
    assign chip_ready = pll_locked & por_rst_n;

endmodule 