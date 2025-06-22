module kan_tda_fpga_top #(
    parameter NUM_KAN_PES = 64,
    parameter NUM_TDA_UNITS = 8,
    parameter DATA_WIDTH = 16,
    parameter AXI_DATA_WIDTH = 256,
    parameter AXI_ADDR_WIDTH = 32
)(
    input wire sys_clk_p,
    input wire sys_clk_n,
    input wire sys_rst_n,
    
    input wire [AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input wire [7:0] s_axi_awlen,
    input wire [2:0] s_axi_awsize,
    input wire [1:0] s_axi_awburst,
    input wire s_axi_awvalid,
    output wire s_axi_awready,
    
    input wire [AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input wire [AXI_DATA_WIDTH/8-1:0] s_axi_wstrb,
    input wire s_axi_wlast,
    input wire s_axi_wvalid,
    output wire s_axi_wready,
    
    output wire [1:0] s_axi_bresp,
    output wire s_axi_bvalid,
    input wire s_axi_bready,
    
    input wire [AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input wire [7:0] s_axi_arlen,
    input wire [2:0] s_axi_arsize,
    input wire [1:0] s_axi_arburst,
    input wire s_axi_arvalid,
    output wire s_axi_arready,
    
    output wire [AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output wire [1:0] s_axi_rresp,
    output wire s_axi_rlast,
    output wire s_axi_rvalid,
    input wire s_axi_rready,
    
    output wire [13:0] ddr4_addr,
    output wire [2:0] ddr4_ba,
    output wire [1:0] ddr4_bg,
    output wire ddr4_ck_p,
    output wire ddr4_ck_n,
    output wire ddr4_cke,
    output wire ddr4_cs_n,
    inout wire [63:0] ddr4_dq,
    inout wire [7:0] ddr4_dqs_p,
    inout wire [7:0] ddr4_dqs_n,
    output wire ddr4_odt,
    output wire ddr4_ras_n,
    output wire ddr4_cas_n,
    output wire ddr4_we_n,
    output wire ddr4_reset_n,
    
    output wire [7:0] status_leds,
    input wire [3:0] user_buttons,
    output wire processing_complete,
    output wire error_flag
);

    wire sys_clk, clk_200mhz, clk_400mhz;
    wire locked, rst_n;
    wire ui_clk, ui_rst_n, init_calib_complete;
    
    wire [DATA_WIDTH-1:0] kan_input_data [0:NUM_KAN_PES-1];
    wire [DATA_WIDTH-1:0] kan_output_data [0:NUM_KAN_PES-1];
    wire [NUM_KAN_PES-1:0] kan_pe_ready;
    wire kan_computation_done;
    
    wire [DATA_WIDTH-1:0] tda_simplex_data [0:NUM_TDA_UNITS-1][0:511];
    wire [DATA_WIDTH-1:0] tda_persistence_pairs [0:NUM_TDA_UNITS-1][0:1023][1:0];
    wire [NUM_TDA_UNITS-1:0] tda_computation_complete;
    
    wire [31:0] control_registers [0:15];
    wire [31:0] status_registers [0:15];
    
    wire [AXI_DATA_WIDTH-1:0] mem_axi_wdata, mem_axi_rdata;
    wire [AXI_ADDR_WIDTH-1:0] mem_axi_awaddr, mem_axi_araddr;
    wire mem_axi_awvalid, mem_axi_awready, mem_axi_wvalid, mem_axi_wready;
    wire mem_axi_bvalid, mem_axi_bready, mem_axi_arvalid, mem_axi_arready;
    wire mem_axi_rvalid, mem_axi_rready;
    
    IBUFGDS clk_buf_inst (
        .O(sys_clk),
        .I(sys_clk_p),
        .IB(sys_clk_n)
    );
    
    clk_wiz_0 clk_gen_inst (
        .clk_in1(sys_clk),
        .reset(~sys_rst_n),
        .clk_out1(clk_200mhz),
        .clk_out2(clk_400mhz),
        .locked(locked)
    );
    
    assign rst_n = sys_rst_n & locked;
    
    ddr4_controller #(
        .DATA_WIDTH(512),
        .ADDR_WIDTH(28),
        .BURST_LENGTH(8),
        .NUM_BANKS(16)
    ) ddr4_ctrl_inst (
        .clk(clk_400mhz),
        .rst_n(rst_n),
        .ui_clk(ui_clk),
        .ui_rst_n(ui_rst_n),
        .app_addr(mem_axi_awaddr[27:0]),
        .app_cmd(3'b000),
        .app_en(mem_axi_awvalid),
        .app_wdf_data(mem_axi_wdata),
        .app_wdf_wren(mem_axi_wvalid),
        .app_wdf_end(1'b1),
        .app_wdf_mask({64{1'b0}}),
        .app_rdy(mem_axi_awready),
        .app_wdf_rdy(mem_axi_wready),
        .app_rd_data(mem_axi_rdata),
        .app_rd_data_valid(mem_axi_rvalid),
        .app_rd_data_end(),
        .init_calib_complete(init_calib_complete),
        .ddr4_ck_p(ddr4_ck_p),
        .ddr4_ck_n(ddr4_ck_n),
        .ddr4_addr(ddr4_addr),
        .ddr4_ba(ddr4_ba),
        .ddr4_bg(ddr4_bg),
        .ddr4_cke(ddr4_cke),
        .ddr4_cs_n(ddr4_cs_n),
        .ddr4_dm_n(),
        .ddr4_dq(ddr4_dq),
        .ddr4_dqs_p(ddr4_dqs_p),
        .ddr4_dqs_n(ddr4_dqs_n),
        .ddr4_odt(ddr4_odt),
        .ddr4_ras_n(ddr4_ras_n),
        .ddr4_cas_n(ddr4_cas_n),
        .ddr4_we_n(ddr4_we_n),
        .ddr4_reset_n(ddr4_reset_n),
        .ddr4_act_n()
    );
    
    axi4_interconnect #(
        .DATA_WIDTH(AXI_DATA_WIDTH),
        .ADDR_WIDTH(AXI_ADDR_WIDTH),
        .ID_WIDTH(4),
        .NUM_MASTERS(1),
        .NUM_SLAVES(4)
    ) axi_interconnect_inst (
        .aclk(ui_clk),
        .aresetn(ui_rst_n),
        .m_axi_awvalid(s_axi_awvalid),
        .m_axi_awready(s_axi_awready),
        .m_axi_awaddr(s_axi_awaddr),
        .m_axi_awid(4'b0),
        .m_axi_awlen(s_axi_awlen),
        .m_axi_awsize(s_axi_awsize),
        .m_axi_awburst(s_axi_awburst),
        .m_axi_wvalid(s_axi_wvalid),
        .m_axi_wready(s_axi_wready),
        .m_axi_wdata(s_axi_wdata),
        .m_axi_wstrb(s_axi_wstrb),
        .m_axi_wlast(s_axi_wlast),
        .m_axi_bvalid(s_axi_bvalid),
        .m_axi_bready(s_axi_bready),
        .m_axi_bresp(s_axi_bresp),
        .m_axi_bid(),
        .m_axi_arvalid(s_axi_arvalid),
        .m_axi_arready(s_axi_arready),
        .m_axi_araddr(s_axi_araddr),
        .m_axi_arid(4'b0),
        .m_axi_arlen(s_axi_arlen),
        .m_axi_arsize(s_axi_arsize),
        .m_axi_arburst(s_axi_arburst),
        .m_axi_rvalid(s_axi_rvalid),
        .m_axi_rready(s_axi_rready),
        .m_axi_rdata(s_axi_rdata),
        .m_axi_rresp(s_axi_rresp),
        .m_axi_rid(),
        .m_axi_rlast(s_axi_rlast),
        .s_axi_awvalid({3'b0, mem_axi_awvalid}),
        .s_axi_awready({3'b0, mem_axi_awready}),
        .s_axi_awaddr({96'b0, mem_axi_awaddr}),
        .s_axi_awid(16'b0),
        .s_axi_awlen(32'b0),
        .s_axi_awsize(12'b0),
        .s_axi_awburst(8'b0),
        .s_axi_wvalid({3'b0, mem_axi_wvalid}),
        .s_axi_wready({3'b0, mem_axi_wready}),
        .s_axi_wdata({768'b0, mem_axi_wdata}),
        .s_axi_wstrb(128'b0),
        .s_axi_wlast(4'b0),
        .s_axi_bvalid({3'b0, mem_axi_bvalid}),
        .s_axi_bready({3'b0, mem_axi_bready}),
        .s_axi_bresp(8'b0),
        .s_axi_bid(16'b0),
        .s_axi_arvalid({3'b0, mem_axi_arvalid}),
        .s_axi_arready({3'b0, mem_axi_arready}),
        .s_axi_araddr({96'b0, mem_axi_araddr}),
        .s_axi_arid(16'b0),
        .s_axi_arlen(32'b0),
        .s_axi_arsize(12'b0),
        .s_axi_arburst(8'b0),
        .s_axi_rvalid({3'b0, mem_axi_rvalid}),
        .s_axi_rready({3'b0, mem_axi_rready}),
        .s_axi_rdata({768'b0, mem_axi_rdata}),
        .s_axi_rresp(8'b0),
        .s_axi_rid(16'b0),
        .s_axi_rlast(4'b0)
    );
    
    genvar i;
    generate
        for (i = 0; i < NUM_KAN_PES; i = i + 1) begin : kan_pe_gen
            kan_processing_element #(
                .DATA_WIDTH(DATA_WIDTH),
                .SPLINE_ORDER(3),
                .GRID_SIZE(32)
            ) kan_pe_inst (
                .clk(ui_clk),
                .rst_n(ui_rst_n),
                .enable(control_registers[0][i]),
                .input_data(kan_input_data[i]),
                .spline_coeffs(32'h0),
                .knot_vectors(64'h0),
                .update_weights(control_registers[1][0]),
                .output_data(kan_output_data[i]),
                .output_valid(kan_pe_ready[i])
            );
        end
    endgenerate
    
    generate
        for (i = 0; i < NUM_TDA_UNITS; i = i + 1) begin : tda_unit_gen
            tda_acceleration_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .ADDR_WIDTH(16)
            ) tda_unit_inst (
                .clk(ui_clk),
                .rst_n(ui_rst_n),
                .enable(control_registers[2][i]),
                .simplex_data(tda_simplex_data[i][0]),
                .simplex_addr(16'h0),
                .simplex_valid(1'b1),
                .compute_persistence(control_registers[3][i]),
                .persistence_pairs(tda_persistence_pairs[i]),
                .num_pairs(),
                .computation_complete(tda_computation_complete[i])
            );
        end
    endgenerate
    
    always @(posedge ui_clk or negedge ui_rst_n) begin
        if (!ui_rst_n) begin
            for (integer j = 0; j < 16; j = j + 1) begin
                control_registers[j] <= 0;
                status_registers[j] <= 0;
            end
        end else begin
            control_registers[0] <= user_buttons[0] ? 32'hFFFFFFFF : 32'h0;
            control_registers[1][0] <= user_buttons[1];
            control_registers[2] <= user_buttons[2] ? 32'hFFFFFFFF : 32'h0;
            control_registers[3] <= user_buttons[3] ? 32'hFFFFFFFF : 32'h0;
            
            status_registers[0] <= {16'b0, kan_pe_ready};
            status_registers[1] <= {24'b0, tda_computation_complete};
            status_registers[2] <= {31'b0, init_calib_complete};
            status_registers[3] <= {31'b0, kan_computation_done};
        end
    end
    
    assign kan_computation_done = &kan_pe_ready;
    assign processing_complete = kan_computation_done & (&tda_computation_complete);
    assign error_flag = ~init_calib_complete;
    assign status_leds = {processing_complete, error_flag, 2'b0, user_buttons};

endmodule 