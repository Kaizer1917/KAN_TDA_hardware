module noc_router #(
    parameter DATA_WIDTH = 256,
    parameter ADDR_WIDTH = 16,
    parameter NUM_PORTS = 5,
    parameter BUFFER_DEPTH = 8,
    parameter ROUTER_ID = 0
)(
    input wire clk,
    input wire rst_n,
    input wire [NUM_PORTS-1:0] port_valid_in,
    input wire [DATA_WIDTH-1:0] data_in [0:NUM_PORTS-1],
    input wire [ADDR_WIDTH-1:0] dest_addr_in [0:NUM_PORTS-1],
    input wire [2:0] packet_type_in [0:NUM_PORTS-1],
    output reg [NUM_PORTS-1:0] port_ready_out,
    output reg [NUM_PORTS-1:0] port_valid_out,
    output reg [DATA_WIDTH-1:0] data_out [0:NUM_PORTS-1],
    output reg [ADDR_WIDTH-1:0] dest_addr_out [0:NUM_PORTS-1],
    output reg [2:0] packet_type_out [0:NUM_PORTS-1],
    input wire [NUM_PORTS-1:0] port_ready_in
);

    localparam NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3, LOCAL = 4;
    
    reg [DATA_WIDTH-1:0] input_buffers [0:NUM_PORTS-1][0:BUFFER_DEPTH-1];
    reg [ADDR_WIDTH-1:0] addr_buffers [0:NUM_PORTS-1][0:BUFFER_DEPTH-1];
    reg [2:0] type_buffers [0:NUM_PORTS-1][0:BUFFER_DEPTH-1];
    reg [3:0] buffer_head [0:NUM_PORTS-1];
    reg [3:0] buffer_tail [0:NUM_PORTS-1];
    reg [3:0] buffer_count [0:NUM_PORTS-1];
    
    reg [2:0] routing_decision [0:NUM_PORTS-1];
    reg [2:0] arbitration_grant;
    reg [2:0] current_port;
    reg [1:0] arbitration_state;
    
    wire [7:0] dest_x [0:NUM_PORTS-1];
    wire [7:0] dest_y [0:NUM_PORTS-1];
    wire [7:0] router_x = ROUTER_ID[7:0];
    wire [7:0] router_y = ROUTER_ID[15:8];
    
    genvar i;
    generate
        for (i = 0; i < NUM_PORTS; i = i + 1) begin : port_gen
            assign dest_x[i] = dest_addr_in[i][7:0];
            assign dest_y[i] = dest_addr_in[i][15:8];
        end
    endgenerate
    
    function [2:0] xy_routing;
        input [7:0] curr_x, curr_y, dest_x, dest_y;
        begin
            if (dest_x > curr_x)
                xy_routing = EAST;
            else if (dest_x < curr_x)
                xy_routing = WEST;
            else if (dest_y > curr_y)
                xy_routing = NORTH;
            else if (dest_y < curr_y)
                xy_routing = SOUTH;
            else
                xy_routing = LOCAL;
        end
    endfunction
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer j = 0; j < NUM_PORTS; j = j + 1) begin
                buffer_head[j] <= 0;
                buffer_tail[j] <= 0;
                buffer_count[j] <= 0;
                port_ready_out[j] <= 1;
            end
            arbitration_state <= 0;
            current_port <= 0;
        end else begin
            for (integer j = 0; j < NUM_PORTS; j = j + 1) begin
                if (port_valid_in[j] && port_ready_out[j] && buffer_count[j] < BUFFER_DEPTH) begin
                    input_buffers[j][buffer_tail[j]] <= data_in[j];
                    addr_buffers[j][buffer_tail[j]] <= dest_addr_in[j];
                    type_buffers[j][buffer_tail[j]] <= packet_type_in[j];
                    buffer_tail[j] <= (buffer_tail[j] + 1) % BUFFER_DEPTH;
                    buffer_count[j] <= buffer_count[j] + 1;
                end
                
                if (buffer_count[j] >= BUFFER_DEPTH - 1)
                    port_ready_out[j] <= 0;
                else
                    port_ready_out[j] <= 1;
                
                routing_decision[j] <= xy_routing(router_x, router_y, dest_x[j], dest_y[j]);
            end
            
            case (arbitration_state)
                2'b00: begin
                    current_port <= 0;
                    arbitration_state <= 2'b01;
                end
                2'b01: begin
                    if (current_port < NUM_PORTS) begin
                        if (buffer_count[current_port] > 0) begin
                            arbitration_grant <= routing_decision[current_port];
                            if (port_ready_in[routing_decision[current_port]]) begin
                                data_out[routing_decision[current_port]] <= input_buffers[current_port][buffer_head[current_port]];
                                dest_addr_out[routing_decision[current_port]] <= addr_buffers[current_port][buffer_head[current_port]];
                                packet_type_out[routing_decision[current_port]] <= type_buffers[current_port][buffer_head[current_port]];
                                port_valid_out[routing_decision[current_port]] <= 1;
                                buffer_head[current_port] <= (buffer_head[current_port] + 1) % BUFFER_DEPTH;
                                buffer_count[current_port] <= buffer_count[current_port] - 1;
                            end
                        end
                        current_port <= current_port + 1;
                    end else begin
                        arbitration_state <= 2'b10;
                    end
                end
                2'b10: begin
                    for (integer k = 0; k < NUM_PORTS; k = k + 1) begin
                        if (!port_ready_in[k])
                            port_valid_out[k] <= 0;
                    end
                    arbitration_state <= 2'b00;
                end
            endcase
        end
    end

endmodule 