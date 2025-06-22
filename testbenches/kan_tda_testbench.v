module kan_tda_testbench;

    parameter DATA_WIDTH = 16;
    parameter ADDR_WIDTH = 12;
    parameter NUM_KAN_CORES = 16;
    parameter NUM_TDA_UNITS = 4;
    parameter CLK_PERIOD = 10;

    reg clk;
    reg rst_n;
    reg enable;
    reg [DATA_WIDTH-1:0] input_data [0:255];
    reg [31:0] control_register;
    reg [ADDR_WIDTH-1:0] memory_addr;
    reg memory_we;
    reg [DATA_WIDTH-1:0] memory_data_in;
    
    wire [DATA_WIDTH-1:0] memory_data_out;
    wire [DATA_WIDTH-1:0] output_data [0:255];
    wire computation_complete;
    wire [15:0] power_status;

    integer i, j, test_cycle;
    reg [DATA_WIDTH-1:0] expected_output;
    reg test_passed;

    kan_tda_asic_top #(
        .NUM_KAN_CORES(NUM_KAN_CORES),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .NUM_TDA_UNITS(NUM_TDA_UNITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .input_data(input_data),
        .control_register(control_register),
        .memory_addr(memory_addr),
        .memory_we(memory_we),
        .memory_data_in(memory_data_in),
        .memory_data_out(memory_data_out),
        .output_data(output_data),
        .computation_complete(computation_complete),
        .power_status(power_status)
    );

    always #(CLK_PERIOD/2) clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        enable = 0;
        control_register = 0;
        memory_addr = 0;
        memory_we = 0;
        memory_data_in = 0;
        test_cycle = 0;
        test_passed = 1;
        
        for (i = 0; i < 256; i = i + 1) begin
            input_data[i] = 0;
        end

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        $display("Starting KAN_TDA Hardware Verification Tests");
        $display("============================================");

        test_kan_processing();
        test_tda_acceleration();
        test_power_management();
        test_integrated_system();

        if (test_passed) begin
            $display("All tests PASSED!");
        end else begin
            $display("Some tests FAILED!");
        end

        $finish;
    end

    task test_kan_processing;
        begin
            $display("Test 1: KAN Processing Elements");
            
            enable = 1;
            control_register[15:0] = 16'hFFFF;
            
            for (i = 0; i < 16; i = i + 1) begin
                for (j = 0; j < 16; j = j + 1) begin
                    input_data[i*16 + j] = i * 16 + j + 1;
                end
            end
            
            memory_addr = 12'h000;
            memory_we = 1;
            for (i = 0; i < 512; i = i + 1) begin
                memory_data_in = $random & 16'hFFFF;
                #CLK_PERIOD;
                memory_addr = memory_addr + 1;
            end
            memory_we = 0;
            
            #(CLK_PERIOD * 100);
            
            if (computation_complete) begin
                $display("KAN processing test PASSED");
            end else begin
                $display("KAN processing test FAILED");
                test_passed = 0;
            end
            
            enable = 0;
            #(CLK_PERIOD * 10);
        end
    endtask

    task test_tda_acceleration;
        begin
            $display("Test 2: TDA Acceleration Units");
            
            enable = 1;
            control_register[19:16] = 4'hF;
            
            for (i = 0; i < 4; i = i + 1) begin
                input_data[i] = i + 1;
            end
            
            memory_addr = 12'h100;
            memory_we = 1;
            for (i = 0; i < 256; i = i + 1) begin
                memory_data_in = (i % 2 == 0) ? 16'h0001 : 16'h0000;
                #CLK_PERIOD;
                memory_addr = memory_addr + 1;
            end
            memory_we = 0;
            
            control_register[19:16] = 4'hF;
            #(CLK_PERIOD * 200);
            
            if (computation_complete) begin
                $display("TDA acceleration test PASSED");
            end else begin
                $display("TDA acceleration test FAILED");
                test_passed = 0;
            end
            
            enable = 0;
            control_register = 0;
            #(CLK_PERIOD * 10);
        end
    endtask

    task test_power_management;
        begin
            $display("Test 3: Power Management");
            
            enable = 1;
            control_register[7:0] = 8'h0F;
            
            #(CLK_PERIOD * 50);
            
            if (power_status < 16'd150) begin
                $display("Power management test PASSED - Power: %d mW", power_status);
            end else begin
                $display("Power management test FAILED - Power: %d mW", power_status);
                test_passed = 0;
            end
            
            control_register = 0;
            enable = 0;
            #(CLK_PERIOD * 10);
        end
    endtask

    task test_integrated_system;
        begin
            $display("Test 4: Integrated System Performance");
            
            enable = 1;
            control_register = 32'h000FFFFF;
            
            for (i = 0; i < 256; i = i + 1) begin
                input_data[i] = $sin(i * 3.14159 / 128) * 32767;
            end
            
            memory_addr = 12'h000;
            memory_we = 1;
            for (i = 0; i < 1024; i = i + 1) begin
                memory_data_in = $random & 16'hFFFF;
                #CLK_PERIOD;
                memory_addr = memory_addr + 1;
            end
            memory_we = 0;
            
            #(CLK_PERIOD * 500);
            
            if (computation_complete && (power_status < 16'd200)) begin
                $display("Integrated system test PASSED");
                $display("Final power consumption: %d mW", power_status);
            end else begin
                $display("Integrated system test FAILED");
                test_passed = 0;
            end
            
            enable = 0;
            control_register = 0;
        end
    endtask

    always @(posedge clk) begin
        if (enable) begin
            test_cycle = test_cycle + 1;
            if (test_cycle % 1000 == 0) begin
                $display("Test cycle: %d, Power: %d mW", test_cycle, power_status);
            end
        end
    end

endmodule 