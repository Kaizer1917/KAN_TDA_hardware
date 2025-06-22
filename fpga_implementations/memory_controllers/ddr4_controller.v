module ddr4_controller #(
    parameter DATA_WIDTH = 512,
    parameter ADDR_WIDTH = 28,
    parameter BURST_LENGTH = 8,
    parameter NUM_BANKS = 16
)(
    input wire clk,
    input wire rst_n,
    input wire ui_clk,
    input wire ui_rst_n,
    input wire [ADDR_WIDTH-1:0] app_addr,
    input wire [2:0] app_cmd,
    input wire app_en,
    input wire [DATA_WIDTH-1:0] app_wdf_data,
    input wire app_wdf_wren,
    input wire app_wdf_end,
    input wire [DATA_WIDTH/8-1:0] app_wdf_mask,
    output reg app_rdy,
    output reg app_wdf_rdy,
    output reg [DATA_WIDTH-1:0] app_rd_data,
    output reg app_rd_data_valid,
    output reg app_rd_data_end,
    output reg init_calib_complete,
    
    output wire ddr4_ck_p,
    output wire ddr4_ck_n,
    output wire [13:0] ddr4_addr,
    output wire [2:0] ddr4_ba,
    output wire [1:0] ddr4_bg,
    output wire ddr4_cke,
    output wire ddr4_cs_n,
    output wire ddr4_dm_n,
    inout wire [63:0] ddr4_dq,
    inout wire [7:0] ddr4_dqs_p,
    inout wire [7:0] ddr4_dqs_n,
    output wire ddr4_odt,
    output wire ddr4_ras_n,
    output wire ddr4_cas_n,
    output wire ddr4_we_n,
    output wire ddr4_reset_n,
    output wire ddr4_act_n
);

    localparam CMD_WRITE = 3'b000;
    localparam CMD_READ = 3'b001;
    localparam CMD_REFRESH = 3'b010;
    
    reg [3:0] state;
    reg [15:0] refresh_counter;
    reg [7:0] cas_latency;
    reg [7:0] write_latency;
    reg [ADDR_WIDTH-1:0] cmd_queue_addr [0:15];
    reg [2:0] cmd_queue_cmd [0:15];
    reg [DATA_WIDTH-1:0] write_queue_data [0:15];
    reg [DATA_WIDTH/8-1:0] write_queue_mask [0:15];
    reg [3:0] cmd_queue_head, cmd_queue_tail;
    reg [3:0] write_queue_head, write_queue_tail;
    reg [3:0] read_queue_head, read_queue_tail;
    reg [DATA_WIDTH-1:0] read_queue_data [0:15];
    
    reg [13:0] row_addr;
    reg [9:0] col_addr;
    reg [2:0] bank_addr;
    reg [1:0] bank_group;
    reg ras_n, cas_n, we_n;
    reg [7:0] burst_counter;
    reg [15:0] timing_counter;
    
    wire [DATA_WIDTH-1:0] phy_rd_data;
    wire phy_rd_data_valid;
    reg [DATA_WIDTH-1:0] phy_wr_data;
    reg phy_wr_en;
    reg [DATA_WIDTH/8-1:0] phy_wr_mask;
    
    assign ddr4_addr = row_addr;
    assign ddr4_ba = bank_addr;
    assign ddr4_bg = bank_group;
    assign ddr4_ras_n = ras_n;
    assign ddr4_cas_n = cas_n;
    assign ddr4_we_n = we_n;
    assign ddr4_reset_n = rst_n;
    assign ddr4_act_n = ~(state == 4'b0010);
    
    always @(posedge ui_clk or negedge ui_rst_n) begin
        if (!ui_rst_n) begin
            state <= 4'b0000;
            refresh_counter <= 0;
            cas_latency <= 14;
            write_latency <= 12;
            cmd_queue_head <= 0;
            cmd_queue_tail <= 0;
            write_queue_head <= 0;
            write_queue_tail <= 0;
            read_queue_head <= 0;
            read_queue_tail <= 0;
            app_rdy <= 0;
            app_wdf_rdy <= 0;
            app_rd_data_valid <= 0;
            init_calib_complete <= 0;
            timing_counter <= 0;
            burst_counter <= 0;
        end else begin
            refresh_counter <= refresh_counter + 1;
            
            if (app_en && app_rdy) begin
                cmd_queue_addr[cmd_queue_tail] <= app_addr;
                cmd_queue_cmd[cmd_queue_tail] <= app_cmd;
                cmd_queue_tail <= (cmd_queue_tail + 1) % 16;
            end
            
            if (app_wdf_wren && app_wdf_rdy) begin
                write_queue_data[write_queue_tail] <= app_wdf_data;
                write_queue_mask[write_queue_tail] <= app_wdf_mask;
                write_queue_tail <= (write_queue_tail + 1) % 16;
            end
            
            case (state)
                4'b0000: begin
                    if (timing_counter < 1000) begin
                        timing_counter <= timing_counter + 1;
                    end else begin
                        init_calib_complete <= 1;
                        app_rdy <= 1;
                        app_wdf_rdy <= 1;
                        state <= 4'b0001;
                    end
                end
                
                4'b0001: begin
                    if (refresh_counter >= 7800) begin
                        state <= 4'b0110;
                        refresh_counter <= 0;
                        app_rdy <= 0;
                    end else if (cmd_queue_head != cmd_queue_tail) begin
                        row_addr <= cmd_queue_addr[cmd_queue_head][27:14];
                        col_addr <= cmd_queue_addr[cmd_queue_head][13:4];
                        bank_addr <= cmd_queue_addr[cmd_queue_head][3:1];
                        bank_group <= cmd_queue_addr[cmd_queue_head][1:0];
                        
                        if (cmd_queue_cmd[cmd_queue_head] == CMD_READ) begin
                            state <= 4'b0010;
                        end else if (cmd_queue_cmd[cmd_queue_head] == CMD_WRITE) begin
                            state <= 4'b0100;
                        end
                        timing_counter <= 0;
                    end
                end
                
                4'b0010: begin
                    ras_n <= 0;
                    cas_n <= 1;
                    we_n <= 1;
                    if (timing_counter >= 18) begin
                        state <= 4'b0011;
                        timing_counter <= 0;
                    end else begin
                        timing_counter <= timing_counter + 1;
                    end
                end
                
                4'b0011: begin
                    ras_n <= 1;
                    cas_n <= 0;
                    we_n <= 1;
                    if (timing_counter >= cas_latency) begin
                        state <= 4'b0111;
                        timing_counter <= 0;
                        burst_counter <= 0;
                    end else begin
                        timing_counter <= timing_counter + 1;
                    end
                end
                
                4'b0100: begin
                    ras_n <= 0;
                    cas_n <= 1;
                    we_n <= 1;
                    if (timing_counter >= 18) begin
                        state <= 4'b0101;
                        timing_counter <= 0;
                    end else begin
                        timing_counter <= timing_counter + 1;
                    end
                end
                
                4'b0101: begin
                    ras_n <= 1;
                    cas_n <= 0;
                    we_n <= 0;
                    phy_wr_data <= write_queue_data[write_queue_head];
                    phy_wr_mask <= write_queue_mask[write_queue_head];
                    phy_wr_en <= 1;
                    if (timing_counter >= write_latency) begin
                        write_queue_head <= (write_queue_head + 1) % 16;
                        cmd_queue_head <= (cmd_queue_head + 1) % 16;
                        state <= 4'b0001;
                        timing_counter <= 0;
                    end else begin
                        timing_counter <= timing_counter + 1;
                    end
                end
                
                4'b0110: begin
                    ras_n <= 0;
                    cas_n <= 0;
                    we_n <= 0;
                    if (timing_counter >= 260) begin
                        state <= 4'b0001;
                        app_rdy <= 1;
                        timing_counter <= 0;
                    end else begin
                        timing_counter <= timing_counter + 1;
                    end
                end
                
                4'b0111: begin
                    if (phy_rd_data_valid) begin
                        read_queue_data[read_queue_tail] <= phy_rd_data;
                        read_queue_tail <= (read_queue_tail + 1) % 16;
                        burst_counter <= burst_counter + 1;
                        if (burst_counter >= BURST_LENGTH - 1) begin
                            cmd_queue_head <= (cmd_queue_head + 1) % 16;
                            state <= 4'b0001;
                        end
                    end
                end
            endcase
            
            if (read_queue_head != read_queue_tail) begin
                app_rd_data <= read_queue_data[read_queue_head];
                app_rd_data_valid <= 1;
                app_rd_data_end <= (read_queue_head == read_queue_tail - 1);
                read_queue_head <= (read_queue_head + 1) % 16;
            end else begin
                app_rd_data_valid <= 0;
                app_rd_data_end <= 0;
            end
        end
    end

endmodule 