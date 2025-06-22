module cache_controller #(
    parameter DATA_WIDTH = 256,
    parameter ADDR_WIDTH = 32,
    parameter CACHE_SIZE = 32768,
    parameter ASSOCIATIVITY = 4,
    parameter BLOCK_SIZE = 64
)(
    input wire clk,
    input wire rst_n,
    input wire [ADDR_WIDTH-1:0] cpu_addr,
    input wire [DATA_WIDTH-1:0] cpu_data_in,
    input wire cpu_we,
    input wire cpu_re,
    output reg [DATA_WIDTH-1:0] cpu_data_out,
    output reg cpu_ready,
    output reg cache_hit,
    
    output reg [ADDR_WIDTH-1:0] mem_addr,
    output reg [DATA_WIDTH-1:0] mem_data_out,
    input wire [DATA_WIDTH-1:0] mem_data_in,
    output reg mem_we,
    output reg mem_re,
    input wire mem_ready
);

    localparam NUM_SETS = CACHE_SIZE / (ASSOCIATIVITY * BLOCK_SIZE);
    localparam SET_BITS = $clog2(NUM_SETS);
    localparam OFFSET_BITS = $clog2(BLOCK_SIZE);
    localparam TAG_BITS = ADDR_WIDTH - SET_BITS - OFFSET_BITS;
    
    reg [DATA_WIDTH-1:0] cache_data [0:NUM_SETS-1][0:ASSOCIATIVITY-1];
    reg [TAG_BITS-1:0] cache_tags [0:NUM_SETS-1][0:ASSOCIATIVITY-1];
    reg [ASSOCIATIVITY-1:0] cache_valid [0:NUM_SETS-1];
    reg [ASSOCIATIVITY-1:0] cache_dirty [0:NUM_SETS-1];
    reg [1:0] lru_counter [0:NUM_SETS-1][0:ASSOCIATIVITY-1];
    
    wire [TAG_BITS-1:0] cpu_tag = cpu_addr[ADDR_WIDTH-1:SET_BITS+OFFSET_BITS];
    wire [SET_BITS-1:0] cpu_set = cpu_addr[SET_BITS+OFFSET_BITS-1:OFFSET_BITS];
    wire [OFFSET_BITS-1:0] cpu_offset = cpu_addr[OFFSET_BITS-1:0];
    
    reg [2:0] state;
    reg [1:0] hit_way;
    reg [1:0] victim_way;
    reg [TAG_BITS-1:0] victim_tag;
    reg victim_dirty;
    reg [15:0] access_counter;
    
    function [1:0] find_lru_way;
        input [SET_BITS-1:0] set_index;
        reg [1:0] lru_way;
        reg [1:0] max_lru;
        integer i;
        begin
            lru_way = 0;
            max_lru = 0;
            for (i = 0; i < ASSOCIATIVITY; i = i + 1) begin
                if (lru_counter[set_index][i] > max_lru) begin
                    max_lru = lru_counter[set_index][i];
                    lru_way = i;
                end
            end
            find_lru_way = lru_way;
        end
    endfunction
    
    function [1:0] check_hit;
        input [SET_BITS-1:0] set_index;
        input [TAG_BITS-1:0] tag;
        reg [1:0] way;
        integer i;
        begin
            way = ASSOCIATIVITY;
            for (i = 0; i < ASSOCIATIVITY; i = i + 1) begin
                if (cache_valid[set_index][i] && cache_tags[set_index][i] == tag) begin
                    way = i;
                end
            end
            check_hit = way;
        end
    endfunction
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < NUM_SETS; i = i + 1) begin
                cache_valid[i] <= 0;
                cache_dirty[i] <= 0;
                for (integer j = 0; j < ASSOCIATIVITY; j = j + 1) begin
                    cache_data[i][j] <= 0;
                    cache_tags[i][j] <= 0;
                    lru_counter[i][j] <= 0;
                end
            end
            state <= 0;
            cpu_ready <= 1;
            cache_hit <= 0;
            mem_we <= 0;
            mem_re <= 0;
            access_counter <= 0;
        end else begin
            access_counter <= access_counter + 1;
            
            case (state)
                3'b000: begin
                    if (cpu_re || cpu_we) begin
                        hit_way <= check_hit(cpu_set, cpu_tag);
                        if (check_hit(cpu_set, cpu_tag) < ASSOCIATIVITY) begin
                            cache_hit <= 1;
                            state <= 3'b001;
                        end else begin
                            cache_hit <= 0;
                            victim_way <= find_lru_way(cpu_set);
                            victim_tag <= cache_tags[cpu_set][find_lru_way(cpu_set)];
                            victim_dirty <= cache_dirty[cpu_set][find_lru_way(cpu_set)];
                            state <= 3'b010;
                        end
                        cpu_ready <= 0;
                    end
                end
                
                3'b001: begin
                    if (cpu_re) begin
                        cpu_data_out <= cache_data[cpu_set][hit_way];
                    end else if (cpu_we) begin
                        cache_data[cpu_set][hit_way] <= cpu_data_in;
                        cache_dirty[cpu_set][hit_way] <= 1;
                    end
                    
                    for (integer j = 0; j < ASSOCIATIVITY; j = j + 1) begin
                        if (j == hit_way) begin
                            lru_counter[cpu_set][j] <= 0;
                        end else begin
                            lru_counter[cpu_set][j] <= lru_counter[cpu_set][j] + 1;
                        end
                    end
                    
                    cpu_ready <= 1;
                    state <= 3'b000;
                end
                
                3'b010: begin
                    if (victim_dirty && cache_valid[cpu_set][victim_way]) begin
                        mem_addr <= {victim_tag, cpu_set, {OFFSET_BITS{1'b0}}};
                        mem_data_out <= cache_data[cpu_set][victim_way];
                        mem_we <= 1;
                        state <= 3'b011;
                    end else begin
                        state <= 3'b100;
                    end
                end
                
                3'b011: begin
                    if (mem_ready) begin
                        mem_we <= 0;
                        state <= 3'b100;
                    end
                end
                
                3'b100: begin
                    mem_addr <= {cpu_tag, cpu_set, {OFFSET_BITS{1'b0}}};
                    mem_re <= 1;
                    state <= 3'b101;
                end
                
                3'b101: begin
                    if (mem_ready) begin
                        cache_data[cpu_set][victim_way] <= mem_data_in;
                        cache_tags[cpu_set][victim_way] <= cpu_tag;
                        cache_valid[cpu_set][victim_way] <= 1;
                        cache_dirty[cpu_set][victim_way] <= 0;
                        mem_re <= 0;
                        
                        if (cpu_we) begin
                            cache_data[cpu_set][victim_way] <= cpu_data_in;
                            cache_dirty[cpu_set][victim_way] <= 1;
                        end else begin
                            cpu_data_out <= mem_data_in;
                        end
                        
                        for (integer j = 0; j < ASSOCIATIVITY; j = j + 1) begin
                            if (j == victim_way) begin
                                lru_counter[cpu_set][j] <= 0;
                            end else begin
                                lru_counter[cpu_set][j] <= lru_counter[cpu_set][j] + 1;
                            end
                        end
                        
                        cpu_ready <= 1;
                        state <= 3'b000;
                    end
                end
            endcase
        end
    end

endmodule 