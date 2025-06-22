module axi4_interconnect #(
    parameter DATA_WIDTH = 256,
    parameter ADDR_WIDTH = 32,
    parameter ID_WIDTH = 4,
    parameter NUM_MASTERS = 4,
    parameter NUM_SLAVES = 8,
    parameter STRB_WIDTH = DATA_WIDTH/8
)(
    input wire aclk,
    input wire aresetn,
    
    input wire [NUM_MASTERS-1:0] m_axi_awvalid,
    output reg [NUM_MASTERS-1:0] m_axi_awready,
    input wire [ADDR_WIDTH-1:0] m_axi_awaddr [0:NUM_MASTERS-1],
    input wire [ID_WIDTH-1:0] m_axi_awid [0:NUM_MASTERS-1],
    input wire [7:0] m_axi_awlen [0:NUM_MASTERS-1],
    input wire [2:0] m_axi_awsize [0:NUM_MASTERS-1],
    input wire [1:0] m_axi_awburst [0:NUM_MASTERS-1],
    
    input wire [NUM_MASTERS-1:0] m_axi_wvalid,
    output reg [NUM_MASTERS-1:0] m_axi_wready,
    input wire [DATA_WIDTH-1:0] m_axi_wdata [0:NUM_MASTERS-1],
    input wire [STRB_WIDTH-1:0] m_axi_wstrb [0:NUM_MASTERS-1],
    input wire [NUM_MASTERS-1:0] m_axi_wlast,
    
    output reg [NUM_MASTERS-1:0] m_axi_bvalid,
    input wire [NUM_MASTERS-1:0] m_axi_bready,
    output reg [1:0] m_axi_bresp [0:NUM_MASTERS-1],
    output reg [ID_WIDTH-1:0] m_axi_bid [0:NUM_MASTERS-1],
    
    input wire [NUM_MASTERS-1:0] m_axi_arvalid,
    output reg [NUM_MASTERS-1:0] m_axi_arready,
    input wire [ADDR_WIDTH-1:0] m_axi_araddr [0:NUM_MASTERS-1],
    input wire [ID_WIDTH-1:0] m_axi_arid [0:NUM_MASTERS-1],
    input wire [7:0] m_axi_arlen [0:NUM_MASTERS-1],
    input wire [2:0] m_axi_arsize [0:NUM_MASTERS-1],
    input wire [1:0] m_axi_arburst [0:NUM_MASTERS-1],
    
    output reg [NUM_MASTERS-1:0] m_axi_rvalid,
    input wire [NUM_MASTERS-1:0] m_axi_rready,
    output reg [DATA_WIDTH-1:0] m_axi_rdata [0:NUM_MASTERS-1],
    output reg [1:0] m_axi_rresp [0:NUM_MASTERS-1],
    output reg [ID_WIDTH-1:0] m_axi_rid [0:NUM_MASTERS-1],
    output reg [NUM_MASTERS-1:0] m_axi_rlast,
    
    output reg [NUM_SLAVES-1:0] s_axi_awvalid,
    input wire [NUM_SLAVES-1:0] s_axi_awready,
    output reg [ADDR_WIDTH-1:0] s_axi_awaddr [0:NUM_SLAVES-1],
    output reg [ID_WIDTH-1:0] s_axi_awid [0:NUM_SLAVES-1],
    output reg [7:0] s_axi_awlen [0:NUM_SLAVES-1],
    output reg [2:0] s_axi_awsize [0:NUM_SLAVES-1],
    output reg [1:0] s_axi_awburst [0:NUM_SLAVES-1],
    
    output reg [NUM_SLAVES-1:0] s_axi_wvalid,
    input wire [NUM_SLAVES-1:0] s_axi_wready,
    output reg [DATA_WIDTH-1:0] s_axi_wdata [0:NUM_SLAVES-1],
    output reg [STRB_WIDTH-1:0] s_axi_wstrb [0:NUM_SLAVES-1],
    output reg [NUM_SLAVES-1:0] s_axi_wlast,
    
    input wire [NUM_SLAVES-1:0] s_axi_bvalid,
    output reg [NUM_SLAVES-1:0] s_axi_bready,
    input wire [1:0] s_axi_bresp [0:NUM_SLAVES-1],
    input wire [ID_WIDTH-1:0] s_axi_bid [0:NUM_SLAVES-1],
    
    output reg [NUM_SLAVES-1:0] s_axi_arvalid,
    input wire [NUM_SLAVES-1:0] s_axi_arready,
    output reg [ADDR_WIDTH-1:0] s_axi_araddr [0:NUM_SLAVES-1],
    output reg [ID_WIDTH-1:0] s_axi_arid [0:NUM_SLAVES-1],
    output reg [7:0] s_axi_arlen [0:NUM_SLAVES-1],
    output reg [2:0] s_axi_arsize [0:NUM_SLAVES-1],
    output reg [1:0] s_axi_arburst [0:NUM_SLAVES-1],
    
    input wire [NUM_SLAVES-1:0] s_axi_rvalid,
    output reg [NUM_SLAVES-1:0] s_axi_rready,
    input wire [DATA_WIDTH-1:0] s_axi_rdata [0:NUM_SLAVES-1],
    input wire [1:0] s_axi_rresp [0:NUM_SLAVES-1],
    input wire [ID_WIDTH-1:0] s_axi_rid [0:NUM_SLAVES-1],
    input wire [NUM_SLAVES-1:0] s_axi_rlast
);

    reg [2:0] aw_arb_grant;
    reg [2:0] ar_arb_grant;
    reg [2:0] w_route [0:NUM_MASTERS-1];
    reg [2:0] r_route [0:NUM_MASTERS-1];
    reg [1:0] arb_state;
    reg [3:0] current_master;
    
    reg [ADDR_WIDTH-1:0] addr_decode_base [0:NUM_SLAVES-1];
    reg [ADDR_WIDTH-1:0] addr_decode_mask [0:NUM_SLAVES-1];
    
    initial begin
        addr_decode_base[0] = 32'h0000_0000;
        addr_decode_mask[0] = 32'hF000_0000;
        addr_decode_base[1] = 32'h1000_0000;
        addr_decode_mask[1] = 32'hF000_0000;
        addr_decode_base[2] = 32'h2000_0000;
        addr_decode_mask[2] = 32'hF000_0000;
        addr_decode_base[3] = 32'h3000_0000;
        addr_decode_mask[3] = 32'hF000_0000;
        addr_decode_base[4] = 32'h4000_0000;
        addr_decode_mask[4] = 32'hF000_0000;
        addr_decode_base[5] = 32'h5000_0000;
        addr_decode_mask[5] = 32'hF000_0000;
        addr_decode_base[6] = 32'h6000_0000;
        addr_decode_mask[6] = 32'hF000_0000;
        addr_decode_base[7] = 32'h7000_0000;
        addr_decode_mask[7] = 32'hF000_0000;
    end
    
    function [2:0] addr_decode;
        input [ADDR_WIDTH-1:0] addr;
        integer i;
        begin
            addr_decode = 0;
            for (i = 0; i < NUM_SLAVES; i = i + 1) begin
                if ((addr & addr_decode_mask[i]) == addr_decode_base[i]) begin
                    addr_decode = i;
                end
            end
        end
    endfunction
    
    function [2:0] round_robin_arbiter;
        input [NUM_MASTERS-1:0] requests;
        input [2:0] last_grant;
        reg [2:0] grant;
        integer i;
        begin
            grant = 0;
            for (i = 1; i < NUM_MASTERS; i = i + 1) begin
                if (requests[(last_grant + i) % NUM_MASTERS]) begin
                    grant = (last_grant + i) % NUM_MASTERS;
                    i = NUM_MASTERS;
                end
            end
            if (grant == 0 && requests[last_grant])
                grant = last_grant;
            round_robin_arbiter = grant;
        end
    endfunction
    
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            for (integer i = 0; i < NUM_MASTERS; i = i + 1) begin
                m_axi_awready[i] <= 0;
                m_axi_wready[i] <= 0;
                m_axi_bvalid[i] <= 0;
                m_axi_arready[i] <= 0;
                m_axi_rvalid[i] <= 0;
                w_route[i] <= 0;
                r_route[i] <= 0;
            end
            for (integer i = 0; i < NUM_SLAVES; i = i + 1) begin
                s_axi_awvalid[i] <= 0;
                s_axi_wvalid[i] <= 0;
                s_axi_bready[i] <= 0;
                s_axi_arvalid[i] <= 0;
                s_axi_rready[i] <= 0;
            end
            aw_arb_grant <= 0;
            ar_arb_grant <= 0;
            arb_state <= 0;
            current_master <= 0;
        end else begin
            case (arb_state)
                2'b00: begin
                    aw_arb_grant <= round_robin_arbiter(m_axi_awvalid, aw_arb_grant);
                    ar_arb_grant <= round_robin_arbiter(m_axi_arvalid, ar_arb_grant);
                    arb_state <= 2'b01;
                end
                
                2'b01: begin
                    if (m_axi_awvalid[aw_arb_grant]) begin
                        w_route[aw_arb_grant] <= addr_decode(m_axi_awaddr[aw_arb_grant]);
                        s_axi_awvalid[addr_decode(m_axi_awaddr[aw_arb_grant])] <= 1;
                        s_axi_awaddr[addr_decode(m_axi_awaddr[aw_arb_grant])] <= m_axi_awaddr[aw_arb_grant];
                        s_axi_awid[addr_decode(m_axi_awaddr[aw_arb_grant])] <= m_axi_awid[aw_arb_grant];
                        s_axi_awlen[addr_decode(m_axi_awaddr[aw_arb_grant])] <= m_axi_awlen[aw_arb_grant];
                        s_axi_awsize[addr_decode(m_axi_awaddr[aw_arb_grant])] <= m_axi_awsize[aw_arb_grant];
                        s_axi_awburst[addr_decode(m_axi_awaddr[aw_arb_grant])] <= m_axi_awburst[aw_arb_grant];
                        m_axi_awready[aw_arb_grant] <= s_axi_awready[addr_decode(m_axi_awaddr[aw_arb_grant])];
                    end
                    
                    if (m_axi_arvalid[ar_arb_grant]) begin
                        r_route[ar_arb_grant] <= addr_decode(m_axi_araddr[ar_arb_grant]);
                        s_axi_arvalid[addr_decode(m_axi_araddr[ar_arb_grant])] <= 1;
                        s_axi_araddr[addr_decode(m_axi_araddr[ar_arb_grant])] <= m_axi_araddr[ar_arb_grant];
                        s_axi_arid[addr_decode(m_axi_araddr[ar_arb_grant])] <= m_axi_arid[ar_arb_grant];
                        s_axi_arlen[addr_decode(m_axi_araddr[ar_arb_grant])] <= m_axi_arlen[ar_arb_grant];
                        s_axi_arsize[addr_decode(m_axi_araddr[ar_arb_grant])] <= m_axi_arsize[ar_arb_grant];
                        s_axi_arburst[addr_decode(m_axi_araddr[ar_arb_grant])] <= m_axi_arburst[ar_arb_grant];
                        m_axi_arready[ar_arb_grant] <= s_axi_arready[addr_decode(m_axi_araddr[ar_arb_grant])];
                    end
                    arb_state <= 2'b10;
                end
                
                2'b10: begin
                    for (integer i = 0; i < NUM_MASTERS; i = i + 1) begin
                        if (m_axi_wvalid[i]) begin
                            s_axi_wvalid[w_route[i]] <= 1;
                            s_axi_wdata[w_route[i]] <= m_axi_wdata[i];
                            s_axi_wstrb[w_route[i]] <= m_axi_wstrb[i];
                            s_axi_wlast[w_route[i]] <= m_axi_wlast[i];
                            m_axi_wready[i] <= s_axi_wready[w_route[i]];
                        end
                    end
                    
                    for (integer i = 0; i < NUM_SLAVES; i = i + 1) begin
                        for (integer j = 0; j < NUM_MASTERS; j = j + 1) begin
                            if (r_route[j] == i && s_axi_rvalid[i]) begin
                                m_axi_rvalid[j] <= 1;
                                m_axi_rdata[j] <= s_axi_rdata[i];
                                m_axi_rresp[j] <= s_axi_rresp[i];
                                m_axi_rid[j] <= s_axi_rid[i];
                                m_axi_rlast[j] <= s_axi_rlast[i];
                                s_axi_rready[i] <= m_axi_rready[j];
                            end
                            
                            if (w_route[j] == i && s_axi_bvalid[i]) begin
                                m_axi_bvalid[j] <= 1;
                                m_axi_bresp[j] <= s_axi_bresp[i];
                                m_axi_bid[j] <= s_axi_bid[i];
                                s_axi_bready[i] <= m_axi_bready[j];
                            end
                        end
                    end
                    arb_state <= 2'b00;
                end
            endcase
        end
    end

endmodule 