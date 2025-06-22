module kan_processing_element #(
    parameter DATA_WIDTH = 16,
    parameter COEFF_WIDTH = 16,
    parameter GRID_SIZE = 8,
    parameter NUM_INPUTS = 4,
    parameter BRAM_ADDR_WIDTH = 9
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] inputs [0:NUM_INPUTS-1],
    input wire [BRAM_ADDR_WIDTH-1:0] coeff_addr,
    input wire coeff_we,
    input wire [COEFF_WIDTH-1:0] coeff_data_in,
    output reg [DATA_WIDTH-1:0] pe_output,
    output reg output_valid
);

    wire [DATA_WIDTH-1:0] bspline_outputs [0:NUM_INPUTS-1];
    wire bspline_valid [0:NUM_INPUTS-1];
    reg [COEFF_WIDTH-1:0] coefficient_memory [0:511];
    reg [DATA_WIDTH-1:0] accumulator;
    reg [2:0] mac_stage;
    reg [2:0] input_index;

    genvar i;
    generate
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin : bspline_gen
            bspline_evaluator #(
                .DATA_WIDTH(DATA_WIDTH),
                .COEFF_WIDTH(COEFF_WIDTH),
                .GRID_SIZE(GRID_SIZE)
            ) bspline_unit (
                .clk(clk),
                .rst_n(rst_n),
                .enable(enable),
                .input_value(inputs[i]),
                .coefficients(coefficient_memory[i*GRID_SIZE +: GRID_SIZE]),
                .knot_vector(coefficient_memory[(NUM_INPUTS*GRID_SIZE + i*(GRID_SIZE+4)) +: (GRID_SIZE+4)]),
                .output_value(bspline_outputs[i]),
                .valid_out(bspline_valid[i])
            );
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pe_output <= 0;
            output_valid <= 0;
            accumulator <= 0;
            mac_stage <= 0;
            input_index <= 0;
        end else begin
            if (coeff_we) begin
                coefficient_memory[coeff_addr] <= coeff_data_in;
            end
            
            if (enable && (&bspline_valid)) begin
                case (mac_stage)
                    3'b000: begin
                        accumulator <= 0;
                        input_index <= 0;
                        mac_stage <= 3'b001;
                        output_valid <= 0;
                    end
                    3'b001: begin
                        accumulator <= accumulator + bspline_outputs[input_index];
                        if (input_index < NUM_INPUTS - 1) begin
                            input_index <= input_index + 1;
                        end else begin
                            mac_stage <= 3'b010;
                        end
                    end
                    3'b010: begin
                        pe_output <= accumulator;
                        output_valid <= 1;
                        mac_stage <= 3'b000;
                    end
                endcase
            end
        end
    end

endmodule 