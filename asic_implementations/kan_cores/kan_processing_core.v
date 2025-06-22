module kan_processing_core #(
    parameter NUM_PES = 64,
    parameter DATA_WIDTH = 16,
    parameter COEFF_WIDTH = 16,
    parameter GRID_SIZE = 8,
    parameter NUM_INPUTS = 4,
    parameter L2_CACHE_SIZE = 65536
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] core_inputs [0:NUM_PES-1][0:NUM_INPUTS-1],
    input wire [15:0] l2_cache_addr,
    input wire l2_cache_we,
    input wire [DATA_WIDTH-1:0] l2_cache_data_in,
    output wire [DATA_WIDTH-1:0] l2_cache_data_out,
    output wire [DATA_WIDTH-1:0] core_outputs [0:NUM_PES-1],
    output wire core_outputs_valid [0:NUM_PES-1],
    output wire core_ready
);

    wire [DATA_WIDTH-1:0] pe_outputs [0:NUM_PES-1];
    wire pe_outputs_valid [0:NUM_PES-1];
    reg [DATA_WIDTH-1:0] l2_cache [0:L2_CACHE_SIZE-1];
    reg [DATA_WIDTH-1:0] l2_cache_out_reg;
    wire [8:0] pe_coeff_addr [0:NUM_PES-1];
    wire pe_coeff_we [0:NUM_PES-1];
    wire [COEFF_WIDTH-1:0] pe_coeff_data [0:NUM_PES-1];
    
    reg [5:0] pe_enable_counter;
    reg [DATA_WIDTH-1:0] prefetch_buffer [0:255];
    reg [7:0] prefetch_index;
    wire all_pes_ready;

    genvar i;
    generate
        for (i = 0; i < NUM_PES; i = i + 1) begin : pe_gen
            kan_processing_element #(
                .DATA_WIDTH(DATA_WIDTH),
                .COEFF_WIDTH(COEFF_WIDTH),
                .GRID_SIZE(GRID_SIZE),
                .NUM_INPUTS(NUM_INPUTS)
            ) pe_inst (
                .clk(clk),
                .rst_n(rst_n),
                .enable(enable && (pe_enable_counter == i)),
                .inputs(core_inputs[i]),
                .coeff_addr(pe_coeff_addr[i]),
                .coeff_we(pe_coeff_we[i]),
                .coeff_data_in(pe_coeff_data[i]),
                .pe_output(pe_outputs[i]),
                .output_valid(pe_outputs_valid[i])
            );
            
            assign pe_coeff_addr[i] = l2_cache_addr[8:0];
            assign pe_coeff_we[i] = l2_cache_we && (l2_cache_addr[15:9] == i);
            assign pe_coeff_data[i] = l2_cache_data_in;
            assign core_outputs[i] = pe_outputs[i];
            assign core_outputs_valid[i] = pe_outputs_valid[i];
        end
    endgenerate

    assign all_pes_ready = &pe_outputs_valid;
    assign core_ready = all_pes_ready;
    assign l2_cache_data_out = l2_cache_out_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pe_enable_counter <= 0;
            prefetch_index <= 0;
            l2_cache_out_reg <= 0;
        end else begin
            if (l2_cache_we) begin
                l2_cache[l2_cache_addr] <= l2_cache_data_in;
            end else begin
                l2_cache_out_reg <= l2_cache[l2_cache_addr];
            end
            
            if (enable) begin
                pe_enable_counter <= pe_enable_counter + 1;
                if (pe_enable_counter >= NUM_PES - 1) begin
                    pe_enable_counter <= 0;
                end
                
                if (prefetch_index < 255) begin
                    prefetch_buffer[prefetch_index] <= l2_cache[l2_cache_addr + prefetch_index + 1];
                    prefetch_index <= prefetch_index + 1;
                end else begin
                    prefetch_index <= 0;
                end
            end
        end
    end

endmodule 