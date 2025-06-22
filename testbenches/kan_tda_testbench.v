module kan_tda_testbench();

parameter DATA_WIDTH = 32;
parameter MATRIX_SIZE = 256;
parameter NUM_POINTS = 64;
parameter CLOCK_PERIOD = 10;

reg clk;
reg rst_n;
reg enable;

reg [DATA_WIDTH-1:0] input_data;
reg input_valid;
wire input_ready;

wire [DATA_WIDTH-1:0] output_data;
wire output_valid;
reg output_ready;

reg [15:0] config_addr;
reg [DATA_WIDTH-1:0] config_data;
reg config_wr_en;

wire computation_done;
wire error_flag;
wire [7:0] status_reg;

reg [DATA_WIDTH-1:0] test_points [NUM_POINTS-1:0][2:0];
reg [7:0] point_counter;
reg [7:0] test_phase;

integer i, j, k;
integer test_cycles;
integer error_count;

tda_asic_top #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_WIDTH(16),
    .NUM_POINTS(NUM_POINTS),
    .NUM_KAN_CORES(4),
    .NUM_TDA_ENGINES(2),
    .MATRIX_SIZE(64)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .input_data(input_data),
    .input_valid(input_valid),
    .input_ready(input_ready),
    .output_data(output_data),
    .output_valid(output_valid),
    .output_ready(output_ready),
    .config_addr(config_addr),
    .config_data(config_data),
    .config_wr_en(config_wr_en),
    .computation_done(computation_done),
    .error_flag(error_flag),
    .status_reg(status_reg)
);

always #(CLOCK_PERIOD/2) clk = ~clk;

initial begin
    clk = 0;
    rst_n = 0;
    enable = 0;
    input_data = 0;
    input_valid = 0;
    output_ready = 1;
    config_addr = 0;
    config_data = 0;
    config_wr_en = 0;
    point_counter = 0;
    test_phase = 0;
    test_cycles = 0;
    error_count = 0;
    
    for (i = 0; i < NUM_POINTS; i = i + 1) begin
        for (j = 0; j < 3; j = j + 1) begin
            test_points[i][j] = 0;
        end
    end
    
    #(CLOCK_PERIOD * 5);
    rst_n = 1;
    enable = 1;
    
    $display("Starting KAN_TDA Hardware Test");
    
    generate_test_data();
    configure_system();
    run_computation_test();
    run_performance_test();
    run_stress_test();
    
    $display("Test completed with %d errors", error_count);
    $finish;
end

task generate_test_data;
    begin
        $display("Generating test data...");
        for (i = 0; i < NUM_POINTS; i = i + 1) begin
            test_points[i][0] = $random % 32'h1000;
            test_points[i][1] = $random % 32'h1000;
            test_points[i][2] = $random % 32'h1000;
        end
    end
endtask

task configure_system;
    begin
        $display("Configuring system...");
        
        config_addr = 16'hF000;
        config_data = 32'h1;
        config_wr_en = 1;
        @(posedge clk);
        config_wr_en = 0;
        
        for (i = 0; i < 16; i = i + 1) begin
            config_addr = 16'h0000 + i;
            config_data = 32'h1000 + i * 32'h100;
            config_wr_en = 1;
            @(posedge clk);
            config_wr_en = 0;
        end
        
        for (i = 0; i < 16; i = i + 1) begin
            config_addr = 16'h8000 + i;
            config_data = 32'h2000 + i * 32'h80;
            config_wr_en = 1;
            @(posedge clk);
            config_wr_en = 0;
        end
        
        #(CLOCK_PERIOD * 10);
    end
endtask

task run_computation_test;
    begin
        $display("Running computation test...");
        point_counter = 0;
        
        while (point_counter < NUM_POINTS) begin
            if (input_ready) begin
                input_data = test_points[point_counter][point_counter % 3];
                input_valid = 1;
                @(posedge clk);
                input_valid = 0;
                point_counter = point_counter + 1;
            end else begin
                @(posedge clk);
            end
        end
        
        wait (computation_done);
        $display("Computation completed");
        
        test_cycles = 0;
        while (output_valid && test_cycles < 1000) begin
            if (output_ready) begin
                $display("Output[%d]: %h", test_cycles, output_data);
                @(posedge clk);
                test_cycles = test_cycles + 1;
            end else begin
                @(posedge clk);
            end
        end
        
        if (error_flag) begin
            error_count = error_count + 1;
            $display("ERROR: Computation error detected");
        end
    end
endtask

task run_performance_test;
    begin
        $display("Running performance test...");
        test_cycles = 0;
        
        config_addr = 16'hF000;
        config_data = 32'h1;
        config_wr_en = 1;
        @(posedge clk);
        config_wr_en = 0;
        
        point_counter = 0;
        while (point_counter < NUM_POINTS && test_cycles < 10000) begin
            if (input_ready) begin
                input_data = test_points[point_counter][0];
                input_valid = 1;
                @(posedge clk);
                input_valid = 0;
                point_counter = point_counter + 1;
            end
            test_cycles = test_cycles + 1;
            @(posedge clk);
        end
        
        $display("Performance test: %d cycles for %d points", test_cycles, point_counter);
        
        if (test_cycles > 5000) begin
            error_count = error_count + 1;
            $display("ERROR: Performance test failed - too many cycles");
        end
    end
endtask

task run_stress_test;
    begin
        $display("Running stress test...");
        
        for (k = 0; k < 10; k = k + 1) begin
            rst_n = 0;
            #(CLOCK_PERIOD * 2);
            rst_n = 1;
            #(CLOCK_PERIOD * 2);
            
            point_counter = 0;
            while (point_counter < 16) begin
                if (input_ready) begin
                    input_data = $random;
                    input_valid = 1;
                    @(posedge clk);
                    input_valid = 0;
                    point_counter = point_counter + 1;
                end else begin
                    @(posedge clk);
                end
            end
            
            #(CLOCK_PERIOD * 100);
            
            if (error_flag) begin
                error_count = error_count + 1;
                $display("ERROR: Stress test iteration %d failed", k);
            end
        end
        
        $display("Stress test completed");
    end
endtask

always @(posedge clk) begin
    if (rst_n && enable) begin
        if (status_reg == 8'h01) test_phase = 1;
        else if (status_reg == 8'h02) test_phase = 2;
        else if (status_reg == 8'h04) test_phase = 3;
        else if (status_reg == 8'h08) test_phase = 4;
        else if (status_reg == 8'h10) test_phase = 5;
        else if (status_reg == 8'h20) test_phase = 6;
    end
end

endmodule

module kan_processing_element_testbench();

parameter DATA_WIDTH = 32;
parameter COEFF_WIDTH = 24;
parameter NUM_SPLINES = 8;
parameter CLOCK_PERIOD = 10;

reg clk;
reg rst_n;
reg enable;

reg [DATA_WIDTH-1:0] x_input;
reg x_valid;
wire x_ready;

wire [DATA_WIDTH-1:0] y_output;
wire y_valid;
reg y_ready;

reg [4:0] spline_select;
reg [COEFF_WIDTH-1:0] coefficients [NUM_SPLINES-1:0];
reg coeff_valid;

wire busy;
wire overflow_flag;

integer i, test_case;
real input_value, output_value;

kan_processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .COEFF_WIDTH(COEFF_WIDTH),
    .NUM_SPLINES(NUM_SPLINES),
    .PIPELINE_STAGES(2),
    .LUT_DEPTH(256)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .x_input(x_input),
    .x_valid(x_valid),
    .x_ready(x_ready),
    .y_output(y_output),
    .y_valid(y_valid),
    .y_ready(y_ready),
    .spline_select(spline_select),
    .coefficients(coefficients),
    .coeff_valid(coeff_valid),
    .busy(busy),
    .overflow_flag(overflow_flag)
);

always #(CLOCK_PERIOD/2) clk = ~clk;

initial begin
    clk = 0;
    rst_n = 0;
    enable = 0;
    x_input = 0;
    x_valid = 0;
    y_ready = 1;
    spline_select = 0;
    coeff_valid = 0;
    
    for (i = 0; i < NUM_SPLINES; i = i + 1) begin
        coefficients[i] = 24'h100000 + i * 24'h10000;
    end
    
    #(CLOCK_PERIOD * 5);
    rst_n = 1;
    enable = 1;
    coeff_valid = 1;
    
    $display("Starting KAN Processing Element Test");
    
    for (test_case = 0; test_case < 16; test_case = test_case + 1) begin
        x_input = test_case * 32'h1000;
        x_valid = 1;
        
        wait (x_ready);
        @(posedge clk);
        x_valid = 0;
        
        wait (y_valid);
        $display("Test %d: Input=%h, Output=%h", test_case, x_input, y_output);
        
        if (overflow_flag) begin
            $display("WARNING: Overflow detected in test %d", test_case);
        end
        
        @(posedge clk);
    end
    
    $display("KAN Processing Element Test completed");
    $finish;
end

endmodule

module tda_acceleration_testbench();

parameter DATA_WIDTH = 32;
parameter MATRIX_SIZE = 64;
parameter NUM_PE = 4;
parameter CLOCK_PERIOD = 10;

reg clk;
reg rst_n;
reg enable;

reg [DATA_WIDTH-1:0] point_data;
reg point_valid;
wire point_ready;

wire [DATA_WIDTH-1:0] persistence_data;
wire persistence_valid;
reg persistence_ready;

reg [7:0] num_points;
reg [7:0] dimension;
reg [DATA_WIDTH-1:0] epsilon;

wire computation_active;
wire [7:0] progress_counter;
wire error_flag;

integer i, j;
integer test_points_count;
real test_epsilon;

tda_acceleration_unit #(
    .DATA_WIDTH(DATA_WIDTH),
    .MATRIX_SIZE(MATRIX_SIZE),
    .NUM_PE(NUM_PE),
    .FIFO_DEPTH(128),
    .MAX_DIMENSION(3)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .point_data(point_data),
    .point_valid(point_valid),
    .point_ready(point_ready),
    .persistence_data(persistence_data),
    .persistence_valid(persistence_valid),
    .persistence_ready(persistence_ready),
    .num_points(num_points),
    .dimension(dimension),
    .epsilon(epsilon),
    .computation_active(computation_active),
    .progress_counter(progress_counter),
    .error_flag(error_flag)
);

always #(CLOCK_PERIOD/2) clk = ~clk;

initial begin
    clk = 0;
    rst_n = 0;
    enable = 0;
    point_data = 0;
    point_valid = 0;
    persistence_ready = 1;
    num_points = 16;
    dimension = 2;
    epsilon = 32'h1000;
    test_points_count = 0;
    
    #(CLOCK_PERIOD * 5);
    rst_n = 1;
    enable = 1;
    
    $display("Starting TDA Acceleration Test");
    
    while (test_points_count < num_points * dimension) begin
        if (point_ready) begin
            point_data = $random % 32'h10000;
            point_valid = 1;
            @(posedge clk);
            point_valid = 0;
            test_points_count = test_points_count + 1;
            $display("Loaded point data %d: %h", test_points_count, point_data);
        end else begin
            @(posedge clk);
        end
    end
    
    $display("All points loaded, waiting for computation...");
    
    while (computation_active) begin
        $display("Progress: %d%%", progress_counter);
        #(CLOCK_PERIOD * 100);
    end
    
    $display("Computation completed, collecting results...");
    
    i = 0;
    while (persistence_valid && i < 100) begin
        if (persistence_ready) begin
            $display("Persistence pair %d: %h", i, persistence_data);
            i = i + 1;
            @(posedge clk);
        end else begin
            @(posedge clk);
        end
    end
    
    if (error_flag) begin
        $display("ERROR: TDA computation error detected");
    end else begin
        $display("TDA Acceleration Test completed successfully");
    end
    
    $finish;
end

endmodule 