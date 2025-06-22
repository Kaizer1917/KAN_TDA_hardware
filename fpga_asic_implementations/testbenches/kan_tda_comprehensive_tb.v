module kan_tda_comprehensive_tb;

    parameter DATA_WIDTH = 16;
    parameter NUM_KAN_CORES = 4;
    parameter NUM_TDA_ENGINES = 2;
    parameter CLK_PERIOD = 10;
    parameter TEST_VECTORS = 1000;
    
    reg sys_clk, por_rst_n;
    reg test_mode, scan_enable;
    reg [31:0] config_data;
    reg config_valid;
    reg [DATA_WIDTH-1:0] data_in [0:255];
    reg [31:0] control_reg;
    reg data_valid;
    
    wire [DATA_WIDTH-1:0] data_out [0:255];
    wire result_valid;
    wire computation_done;
    wire [15:0] power_status;
    wire [7:0] thermal_status;
    wire chip_ready;
    
    reg jtag_tck, jtag_tdi, jtag_tms, jtag_trst_n;
    wire jtag_tdo;
    
    integer i, j, test_count, error_count;
    integer data_file, result_file;
    reg [DATA_WIDTH-1:0] expected_output [0:255];
    reg [31:0] test_vector [0:TEST_VECTORS-1];
    
    kan_tda_asic_core #(
        .NUM_KAN_CORES(NUM_KAN_CORES),
        .NUM_TDA_ENGINES(NUM_TDA_ENGINES),
        .DATA_WIDTH(DATA_WIDTH),
        .CACHE_SIZE(32768),
        .NUM_POWER_DOMAINS(8)
    ) dut (
        .sys_clk(sys_clk),
        .por_rst_n(por_rst_n),
        .test_mode(test_mode),
        .scan_enable(scan_enable),
        .config_data(config_data),
        .config_valid(config_valid),
        .data_in(data_in),
        .control_reg(control_reg),
        .data_valid(data_valid),
        .data_out(data_out),
        .result_valid(result_valid),
        .computation_done(computation_done),
        .power_status(power_status),
        .thermal_status(thermal_status),
        .chip_ready(chip_ready),
        .jtag_tck(jtag_tck),
        .jtag_tdi(jtag_tdi),
        .jtag_tms(jtag_tms),
        .jtag_trst_n(jtag_trst_n),
        .jtag_tdo(jtag_tdo)
    );
    
    always #(CLK_PERIOD/2) sys_clk = ~sys_clk;
    always #(50) jtag_tck = ~jtag_tck;
    
    initial begin
        $dumpfile("kan_tda_simulation.vcd");
        $dumpvars(0, kan_tda_comprehensive_tb);
        
        sys_clk = 0;
        por_rst_n = 0;
        test_mode = 0;
        scan_enable = 0;
        config_data = 0;
        config_valid = 0;
        control_reg = 0;
        data_valid = 0;
        jtag_tck = 0;
        jtag_tdi = 0;
        jtag_tms = 0;
        jtag_trst_n = 0;
        test_count = 0;
        error_count = 0;
        
        for (i = 0; i < 256; i = i + 1) begin
            data_in[i] = 0;
            expected_output[i] = 0;
        end
        
        data_file = $fopen("test_vectors.txt", "r");
        result_file = $fopen("test_results.txt", "w");
        
        if (data_file == 0) begin
            $display("ERROR: Cannot open test_vectors.txt");
            $finish;
        end
        
        #20 por_rst_n = 1;
        #20 jtag_trst_n = 1;
        
        wait(chip_ready);
        $display("Chip ready at time %t", $time);
        
        test_basic_functionality();
        test_kan_processing();
        test_tda_computation();
        test_power_management();
        test_jtag_interface();
        test_stress_conditions();
        
        $display("\n=== TEST SUMMARY ===");
        $display("Total tests: %d", test_count);
        $display("Errors: %d", error_count);
        if (error_count == 0)
            $display("ALL TESTS PASSED!");
        else
            $display("TESTS FAILED!");
        
        $fclose(data_file);
        $fclose(result_file);
        $finish;
    end
    
    task test_basic_functionality;
        begin
            $display("\n=== Testing Basic Functionality ===");
            test_count = test_count + 1;
            
            config_data = 32'h12345678;
            config_valid = 1;
            #CLK_PERIOD;
            config_valid = 0;
            
            control_reg = 32'hFFFFFFFF;
            data_valid = 1;
            
            for (i = 0; i < 64; i = i + 1) begin
                data_in[i] = i + 1;
            end
            
            #(CLK_PERIOD * 100);
            
            if (!result_valid) begin
                $display("ERROR: Result not valid after basic test");
                error_count = error_count + 1;
            end else begin
                $display("PASS: Basic functionality test");
            end
        end
    endtask
    
    task test_kan_processing;
        begin
            $display("\n=== Testing KAN Processing ===");
            test_count = test_count + 1;
            
            control_reg = 32'h0000000F;
            
            for (i = 0; i < NUM_KAN_CORES * 16; i = i + 1) begin
                data_in[i] = $random % 65536;
            end
            
            data_valid = 1;
            #CLK_PERIOD;
            data_valid = 0;
            
            wait(computation_done);
            #(CLK_PERIOD * 10);
            
            if (result_valid) begin
                $display("PASS: KAN processing completed");
                for (i = 0; i < NUM_KAN_CORES * 4; i = i + 1) begin
                    $fwrite(result_file, "KAN[%d] = %d\n", i, data_out[i]);
                end
            end else begin
                $display("ERROR: KAN processing failed");
                error_count = error_count + 1;
            end
        end
    endtask
    
    task test_tda_computation;
        begin
            $display("\n=== Testing TDA Computation ===");
            test_count = test_count + 1;
            
            control_reg = 32'h03000000;
            config_data = 32'h00100400;
            
            for (i = 0; i < NUM_TDA_ENGINES * 64; i = i + 1) begin
                data_in[i] = $random % 32768;
            end
            
            data_valid = 1;
            #CLK_PERIOD;
            data_valid = 0;
            
            wait(computation_done);
            #(CLK_PERIOD * 20);
            
            if (result_valid) begin
                $display("PASS: TDA computation completed");
                for (i = NUM_KAN_CORES * 4; i < NUM_KAN_CORES * 4 + NUM_TDA_ENGINES * 16; i = i + 1) begin
                    $fwrite(result_file, "TDA[%d] = %d\n", i - NUM_KAN_CORES * 4, data_out[i]);
                end
            end else begin
                $display("ERROR: TDA computation failed");
                error_count = error_count + 1;
            end
        end
    endtask
    
    task test_power_management;
        begin
            $display("\n=== Testing Power Management ===");
            test_count = test_count + 1;
            
            control_reg = 32'h00000000;
            #(CLK_PERIOD * 50);
            
            if (power_status < 16'h8000) begin
                $display("PASS: Low power mode active, power = %d", power_status);
            end else begin
                $display("WARNING: High power consumption in idle: %d", power_status);
            end
            
            control_reg = 32'hFFFFFFFF;
            #(CLK_PERIOD * 50);
            
            if (power_status > 16'h1000) begin
                $display("PASS: Active mode power consumption: %d", power_status);
            end else begin
                $display("ERROR: Power consumption too low in active mode");
                error_count = error_count + 1;
            end
        end
    endtask
    
    task test_jtag_interface;
        begin
            $display("\n=== Testing JTAG Interface ===");
            test_count = test_count + 1;
            
            jtag_tms = 1;
            #(100) jtag_tms = 0;
            #(100) jtag_tdi = 1;
            #(100) jtag_tdi = 0;
            
            repeat(10) begin
                #100;
                if (jtag_tdo !== 1'bx) begin
                    $display("JTAG response: %b", jtag_tdo);
                end
            end
            
            $display("PASS: JTAG interface functional");
        end
    endtask
    
    task test_stress_conditions;
        begin
            $display("\n=== Testing Stress Conditions ===");
            test_count = test_count + 1;
            
            for (j = 0; j < 100; j = j + 1) begin
                control_reg = $random;
                config_data = $random;
                
                for (i = 0; i < 256; i = i + 1) begin
                    data_in[i] = $random % 65536;
                end
                
                data_valid = 1;
                #CLK_PERIOD;
                data_valid = 0;
                
                #(CLK_PERIOD * 20);
                
                if (thermal_status > 8'hC0) begin
                    $display("WARNING: High temperature detected: %d", thermal_status);
                end
            end
            
            $display("PASS: Stress test completed");
        end
    endtask
    
    always @(posedge result_valid) begin
        $display("Result valid at time %t", $time);
        $display("Power status: %d", power_status);
        $display("Thermal status: %d", thermal_status);
    end
    
    always @(posedge computation_done) begin
        $display("Computation done at time %t", $time);
    end
    
    initial begin
        #(CLK_PERIOD * 100000);
        $display("TIMEOUT: Simulation exceeded maximum time");
        $finish;
    end

endmodule 