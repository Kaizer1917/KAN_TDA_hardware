module bspline_evaluator #(
    parameter DATA_WIDTH = 16,
    parameter COEFF_WIDTH = 16,
    parameter GRID_SIZE = 8,
    parameter DEGREE = 3
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] input_value,
    input wire [COEFF_WIDTH-1:0] coefficients [0:GRID_SIZE-1],
    input wire [DATA_WIDTH-1:0] knot_vector [0:GRID_SIZE+DEGREE],
    output reg [DATA_WIDTH-1:0] output_value,
    output reg valid_out
);

    reg [DATA_WIDTH-1:0] normalized_input;
    reg [DATA_WIDTH-1:0] basis_values [0:GRID_SIZE-1];
    reg [DATA_WIDTH-1:0] temp_result;
    reg [2:0] stage_counter;
    reg [3:0] basis_index;
    
    wire [DATA_WIDTH-1:0] knot_diff;
    wire [DATA_WIDTH-1:0] weight_left, weight_right;
    
    assign knot_diff = knot_vector[basis_index + DEGREE] - knot_vector[basis_index];
    assign weight_left = (normalized_input - knot_vector[basis_index]) / knot_diff;
    assign weight_right = (knot_vector[basis_index + DEGREE + 1] - normalized_input) / 
                         (knot_vector[basis_index + DEGREE + 1] - knot_vector[basis_index + 1]);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_value <= 0;
            valid_out <= 0;
            stage_counter <= 0;
            basis_index <= 0;
            temp_result <= 0;
        end else if (enable) begin
            case (stage_counter)
                3'b000: begin
                    normalized_input <= (input_value * GRID_SIZE) >> DATA_WIDTH;
                    stage_counter <= 3'b001;
                    valid_out <= 0;
                end
                3'b001: begin
                    for (integer i = 0; i < GRID_SIZE; i = i + 1) begin
                        basis_values[i] <= (i == normalized_input) ? {DATA_WIDTH{1'b1}} : 0;
                    end
                    stage_counter <= 3'b010;
                end
                3'b010: begin
                    if (basis_index < GRID_SIZE - 1) begin
                        basis_values[basis_index] <= weight_left * basis_values[basis_index] + 
                                                   weight_right * basis_values[basis_index + 1];
                        basis_index <= basis_index + 1;
                    end else begin
                        stage_counter <= 3'b011;
                        basis_index <= 0;
                    end
                end
                3'b011: begin
                    temp_result <= temp_result + (coefficients[basis_index] * basis_values[basis_index]);
                    if (basis_index < GRID_SIZE - 1) begin
                        basis_index <= basis_index + 1;
                    end else begin
                        stage_counter <= 3'b100;
                    end
                end
                3'b100: begin
                    output_value <= temp_result;
                    valid_out <= 1;
                    stage_counter <= 3'b000;
                    temp_result <= 0;
                    basis_index <= 0;
                end
            endcase
        end
    end

endmodule 