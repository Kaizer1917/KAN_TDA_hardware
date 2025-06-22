module tda_asic_top #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 16,
    parameter NUM_POINTS = 1024,
    parameter NUM_KAN_CORES = 16,
    parameter NUM_TDA_ENGINES = 8,
    parameter MATRIX_SIZE = 256
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] input_data,
    input wire input_valid,
    output wire input_ready,
    
    output wire [DATA_WIDTH-1:0] output_data,
    output wire output_valid,
    input wire output_ready,
    
    input wire [ADDR_WIDTH-1:0] config_addr,
    input wire [DATA_WIDTH-1:0] config_data,
    input wire config_wr_en,
    
    output wire computation_done,
    output wire error_flag,
    output wire [7:0] status_reg
);

wire [NUM_KAN_CORES-1:0] kan_core_enable;
wire [NUM_KAN_CORES-1:0] kan_core_ready;
wire [NUM_KAN_CORES-1:0] kan_core_valid;
wire [DATA_WIDTH-1:0] kan_core_output [NUM_KAN_CORES-1:0];

wire [NUM_TDA_ENGINES-1:0] tda_engine_enable;
wire [NUM_TDA_ENGINES-1:0] tda_engine_ready;
wire [NUM_TDA_ENGINES-1:0] tda_engine_valid;
wire [DATA_WIDTH-1:0] tda_engine_output [NUM_TDA_ENGINES-1:0];

wire [DATA_WIDTH-1:0] distance_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
wire distance_matrix_valid;
wire persistence_computation_done;

wire [DATA_WIDTH-1:0] embeddings [NUM_POINTS-1:0];
wire embedding_valid;

wire [11:0] supply_voltage;
wire [15:0] total_current;
wire [9:0] chip_temperature;
wire [NUM_KAN_CORES+NUM_TDA_ENGINES-1:0] domain_enable;
wire thermal_throttle;
wire power_emergency;

control_unit #(
    .NUM_KAN_CORES(NUM_KAN_CORES),
    .NUM_TDA_ENGINES(NUM_TDA_ENGINES),
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(DATA_WIDTH)
) control (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .config_addr(config_addr),
    .config_data(config_data),
    .config_wr_en(config_wr_en),
    .kan_core_enable(kan_core_enable),
    .kan_core_ready(kan_core_ready),
    .kan_core_valid(kan_core_valid),
    .tda_engine_enable(tda_engine_enable),
    .tda_engine_ready(tda_engine_ready),
    .tda_engine_valid(tda_engine_valid),
    .computation_done(computation_done),
    .status_reg(status_reg),
    .error_flag(error_flag)
);

data_path #(
    .DATA_WIDTH(DATA_WIDTH),
    .NUM_POINTS(NUM_POINTS),
    .MATRIX_SIZE(MATRIX_SIZE)
) datapath (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .input_data(input_data),
    .input_valid(input_valid),
    .input_ready(input_ready),
    .output_data(output_data),
    .output_valid(output_valid),
    .output_ready(output_ready),
    .embeddings(embeddings),
    .embedding_valid(embedding_valid),
    .distance_matrix(distance_matrix),
    .distance_matrix_valid(distance_matrix_valid)
);

genvar i;
generate
    for (i = 0; i < NUM_KAN_CORES; i = i + 1) begin : kan_cores
        kan_processing_core #(
            .DATA_WIDTH(DATA_WIDTH),
            .ADDR_WIDTH(ADDR_WIDTH)
        ) kan_core (
            .clk(domain_enable[i] ? clk : 1'b0),
            .rst_n(rst_n),
            .enable(kan_core_enable[i]),
            .x_input(embeddings[i * (NUM_POINTS/NUM_KAN_CORES)]),
            .x_valid(embedding_valid),
            .x_ready(kan_core_ready[i]),
            .y_output(kan_core_output[i]),
            .y_valid(kan_core_valid[i]),
            .y_ready(1'b1),
            .coeff_addr(config_addr),
            .coeff_data(config_data),
            .coeff_wr_en(config_wr_en && (config_addr[15:12] == i)),
            .knot_addr(config_addr),
            .knot_data(config_data),
            .knot_wr_en(config_wr_en && (config_addr[15:12] == (i + 8))),
            .busy(),
            .error()
        );
    end
endgenerate

generate
    for (i = 0; i < NUM_TDA_ENGINES; i = i + 1) begin : tda_engines
        persistence_engine #(
            .DATA_WIDTH(DATA_WIDTH),
            .MATRIX_SIZE(MATRIX_SIZE)
        ) tda_engine (
            .clk(domain_enable[NUM_KAN_CORES + i] ? clk : 1'b0),
            .rst_n(rst_n),
            .enable(tda_engine_enable[i]),
            .distance_matrix_row(distance_matrix[i * (MATRIX_SIZE/NUM_TDA_ENGINES)]),
            .distance_matrix_valid(distance_matrix_valid),
            .persistence_pairs(tda_engine_output[i]),
            .persistence_valid(tda_engine_valid[i]),
            .computation_done(persistence_computation_done)
        );
    end
endgenerate

power_management_unit #(
    .NUM_DOMAINS(NUM_KAN_CORES + NUM_TDA_ENGINES),
    .VOLTAGE_WIDTH(12),
    .CURRENT_WIDTH(16),
    .TEMP_WIDTH(10)
) power_mgmt (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .supply_voltage(supply_voltage),
    .total_current(total_current),
    .chip_temperature(chip_temperature),
    .domain_active({NUM_TDA_ENGINES{tda_engine_enable}, NUM_KAN_CORES{kan_core_enable}}),
    .domain_request({NUM_TDA_ENGINES{1'b1}, NUM_KAN_CORES{1'b1}}),
    .domain_enable(domain_enable),
    .domain_clock_gate(),
    .target_voltage(),
    .frequency_scale(),
    .thermal_throttle(thermal_throttle),
    .power_emergency(power_emergency),
    .dvfs_enable(1'b1),
    .power_budget(16'h8000),
    .current_power(),
    .power_valid()
);

interconnect_network #(
    .DATA_WIDTH(DATA_WIDTH),
    .NUM_KAN_CORES(NUM_KAN_CORES),
    .NUM_TDA_ENGINES(NUM_TDA_ENGINES)
) interconnect (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .kan_outputs(kan_core_output),
    .kan_valid(kan_core_valid),
    .tda_inputs(),
    .tda_valid(),
    .network_congestion()
);

endmodule

module control_unit #(
    parameter NUM_KAN_CORES = 16,
    parameter NUM_TDA_ENGINES = 8,
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [ADDR_WIDTH-1:0] config_addr,
    input wire [DATA_WIDTH-1:0] config_data,
    input wire config_wr_en,
    
    output reg [NUM_KAN_CORES-1:0] kan_core_enable,
    input wire [NUM_KAN_CORES-1:0] kan_core_ready,
    input wire [NUM_KAN_CORES-1:0] kan_core_valid,
    
    output reg [NUM_TDA_ENGINES-1:0] tda_engine_enable,
    input wire [NUM_TDA_ENGINES-1:0] tda_engine_ready,
    input wire [NUM_TDA_ENGINES-1:0] tda_engine_valid,
    
    output reg computation_done,
    output reg [7:0] status_reg,
    output reg error_flag
);

reg [3:0] state;
reg [15:0] cycle_counter;
reg [DATA_WIDTH-1:0] config_registers [15:0];

localparam IDLE = 4'h0;
localparam INIT = 4'h1;
localparam EMBEDDING = 4'h2;
localparam DISTANCE = 4'h3;
localparam PERSISTENCE = 4'h4;
localparam DONE = 4'h5;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        kan_core_enable <= '0;
        tda_engine_enable <= '0;
        computation_done <= 1'b0;
        status_reg <= 8'h00;
        error_flag <= 1'b0;
        cycle_counter <= 16'h0;
        for (int i = 0; i < 16; i = i + 1) begin
            config_registers[i] <= 32'h0;
        end
    end else if (enable) begin
        if (config_wr_en && config_addr[15:12] == 4'hF) begin
            config_registers[config_addr[3:0]] <= config_data;
        end
        
        cycle_counter <= cycle_counter + 1;
        
        case (state)
            IDLE: begin
                status_reg <= 8'h01;
                if (config_registers[0][0]) begin
                    state <= INIT;
                    kan_core_enable <= '0;
                    tda_engine_enable <= '0;
                    computation_done <= 1'b0;
                end
            end
            
            INIT: begin
                status_reg <= 8'h02;
                if (cycle_counter > 16'h0100) begin
                    state <= EMBEDDING;
                    kan_core_enable <= {NUM_KAN_CORES{1'b1}};
                    cycle_counter <= 16'h0;
                end
            end
            
            EMBEDDING: begin
                status_reg <= 8'h04;
                if (&kan_core_valid) begin
                    state <= DISTANCE;
                    kan_core_enable <= '0;
                    cycle_counter <= 16'h0;
                end else if (cycle_counter > 16'h1000) begin
                    error_flag <= 1'b1;
                    state <= IDLE;
                end
            end
            
            DISTANCE: begin
                status_reg <= 8'h08;
                if (cycle_counter > 16'h0200) begin
                    state <= PERSISTENCE;
                    tda_engine_enable <= {NUM_TDA_ENGINES{1'b1}};
                    cycle_counter <= 16'h0;
                end
            end
            
            PERSISTENCE: begin
                status_reg <= 8'h10;
                if (&tda_engine_valid) begin
                    state <= DONE;
                    tda_engine_enable <= '0;
                    computation_done <= 1'b1;
                end else if (cycle_counter > 16'h2000) begin
                    error_flag <= 1'b1;
                    state <= IDLE;
                end
            end
            
            DONE: begin
                status_reg <= 8'h20;
                if (!config_registers[0][0]) begin
                    state <= IDLE;
                    computation_done <= 1'b0;
                end
            end
            
            default: begin
                state <= IDLE;
                error_flag <= 1'b1;
            end
        endcase
    end
end

endmodule

module data_path #(
    parameter DATA_WIDTH = 32,
    parameter NUM_POINTS = 1024,
    parameter MATRIX_SIZE = 256
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] input_data,
    input wire input_valid,
    output reg input_ready,
    
    output reg [DATA_WIDTH-1:0] output_data,
    output reg output_valid,
    input wire output_ready,
    
    output reg [DATA_WIDTH-1:0] embeddings [NUM_POINTS-1:0],
    output reg embedding_valid,
    
    output reg [DATA_WIDTH-1:0] distance_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0],
    output reg distance_matrix_valid
);

reg [DATA_WIDTH-1:0] input_buffer [NUM_POINTS-1:0];
reg [15:0] input_counter;
reg [15:0] output_counter;
reg data_processing_active;

reg [DATA_WIDTH-1:0] intermediate_results [MATRIX_SIZE-1:0];
reg [7:0] processing_stage;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        input_counter <= 16'h0;
        output_counter <= 16'h0;
        input_ready <= 1'b1;
        output_valid <= 1'b0;
        embedding_valid <= 1'b0;
        distance_matrix_valid <= 1'b0;
        data_processing_active <= 1'b0;
        processing_stage <= 8'h0;
        for (int i = 0; i < NUM_POINTS; i = i + 1) begin
            input_buffer[i] <= 32'h0;
            if (i < MATRIX_SIZE) begin
                embeddings[i] <= 32'h0;
                intermediate_results[i] <= 32'h0;
            end
        end
        for (int j = 0; j < MATRIX_SIZE; j = j + 1) begin
            for (int k = 0; k < MATRIX_SIZE; k = k + 1) begin
                distance_matrix[j][k] <= 32'h0;
            end
        end
    end else if (enable) begin
        if (input_valid && input_ready) begin
            input_buffer[input_counter] <= input_data;
            input_counter <= input_counter + 1;
            
            if (input_counter >= NUM_POINTS - 1) begin
                input_ready <= 1'b0;
                data_processing_active <= 1'b1;
                processing_stage <= 8'h1;
            end
        end
        
        if (data_processing_active) begin
            case (processing_stage)
                8'h1: begin
                    for (int m = 0; m < MATRIX_SIZE; m = m + 1) begin
                        embeddings[m] <= input_buffer[m];
                    end
                    embedding_valid <= 1'b1;
                    processing_stage <= 8'h2;
                end
                
                8'h2: begin
                    for (int n = 0; n < MATRIX_SIZE; n = n + 1) begin
                        for (int p = 0; p < MATRIX_SIZE; p = p + 1) begin
                            if (n != p) begin
                                distance_matrix[n][p] <= embeddings[n] > embeddings[p] ? 
                                                      embeddings[n] - embeddings[p] : 
                                                      embeddings[p] - embeddings[n];
                            end else begin
                                distance_matrix[n][p] <= 32'h0;
                            end
                        end
                    end
                    distance_matrix_valid <= 1'b1;
                    processing_stage <= 8'h3;
                end
                
                8'h3: begin
                    output_valid <= 1'b1;
                    output_data <= intermediate_results[output_counter];
                    
                    if (output_ready) begin
                        output_counter <= output_counter + 1;
                        if (output_counter >= MATRIX_SIZE - 1) begin
                            data_processing_active <= 1'b0;
                            processing_stage <= 8'h0;
                            input_ready <= 1'b1;
                            input_counter <= 16'h0;
                            output_counter <= 16'h0;
                            output_valid <= 1'b0;
                        end
                    end
                end
            endcase
        end
    end
end

endmodule

module persistence_engine #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] distance_matrix_row [MATRIX_SIZE-1:0],
    input wire distance_matrix_valid,
    
    output reg [DATA_WIDTH-1:0] persistence_pairs,
    output reg persistence_valid,
    output reg computation_done
);

reg [DATA_WIDTH-1:0] boundary_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
reg [DATA_WIDTH-1:0] reduced_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
reg [7:0] reduction_state;
reg [7:0] current_col;
reg [7:0] current_row;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        persistence_pairs <= 32'h0;
        persistence_valid <= 1'b0;
        computation_done <= 1'b0;
        reduction_state <= 8'h0;
        current_col <= 8'h0;
        current_row <= 8'h0;
        for (int i = 0; i < MATRIX_SIZE; i = i + 1) begin
            for (int j = 0; j < MATRIX_SIZE; j = j + 1) begin
                boundary_matrix[i][j] <= 32'h0;
                reduced_matrix[i][j] <= 32'h0;
            end
        end
    end else if (enable && distance_matrix_valid) begin
        case (reduction_state)
            8'h0: begin
                for (int k = 0; k < MATRIX_SIZE; k = k + 1) begin
                    boundary_matrix[current_row][k] <= distance_matrix_row[k];
                end
                current_row <= current_row + 1;
                if (current_row >= MATRIX_SIZE - 1) begin
                    reduction_state <= 8'h1;
                    current_row <= 8'h0;
                    current_col <= 8'h0;
                end
            end
            
            8'h1: begin
                if (boundary_matrix[current_row][current_col] != 0) begin
                    reduced_matrix[current_row][current_col] <= boundary_matrix[current_row][current_col];
                end else begin
                    reduced_matrix[current_row][current_col] <= 32'h0;
                end
                
                current_col <= current_col + 1;
                if (current_col >= MATRIX_SIZE - 1) begin
                    current_col <= 8'h0;
                    current_row <= current_row + 1;
                    if (current_row >= MATRIX_SIZE - 1) begin
                        reduction_state <= 8'h2;
                        current_row <= 8'h0;
                    end
                end
            end
            
            8'h2: begin
                persistence_pairs <= reduced_matrix[current_row][0];
                persistence_valid <= 1'b1;
                current_row <= current_row + 1;
                
                if (current_row >= MATRIX_SIZE - 1) begin
                    computation_done <= 1'b1;
                    reduction_state <= 8'h0;
                    current_row <= 8'h0;
                end
            end
        endcase
    end
end

endmodule

module interconnect_network #(
    parameter DATA_WIDTH = 32,
    parameter NUM_KAN_CORES = 16,
    parameter NUM_TDA_ENGINES = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] kan_outputs [NUM_KAN_CORES-1:0],
    input wire [NUM_KAN_CORES-1:0] kan_valid,
    
    output reg [DATA_WIDTH-1:0] tda_inputs [NUM_TDA_ENGINES-1:0],
    output reg [NUM_TDA_ENGINES-1:0] tda_valid,
    
    output reg network_congestion
);

reg [3:0] routing_table [NUM_KAN_CORES-1:0];
reg [15:0] traffic_counter;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        traffic_counter <= 16'h0;
        network_congestion <= 1'b0;
        tda_valid <= '0;
        for (int i = 0; i < NUM_KAN_CORES; i = i + 1) begin
            routing_table[i] <= i % NUM_TDA_ENGINES;
        end
        for (int j = 0; j < NUM_TDA_ENGINES; j = j + 1) begin
            tda_inputs[j] <= 32'h0;
        end
    end else if (enable) begin
        traffic_counter <= traffic_counter + 1;
        network_congestion <= (traffic_counter > 16'hF000);
        
        for (int k = 0; k < NUM_KAN_CORES; k = k + 1) begin
            if (kan_valid[k]) begin
                tda_inputs[routing_table[k]] <= kan_outputs[k];
                tda_valid[routing_table[k]] <= 1'b1;
            end
        end
        
        if (traffic_counter[7:0] == 8'h00) begin
            for (int m = 0; m < NUM_KAN_CORES; m = m + 1) begin
                routing_table[m] <= (routing_table[m] + 1) % NUM_TDA_ENGINES;
            end
        end
    end
end

endmodule 