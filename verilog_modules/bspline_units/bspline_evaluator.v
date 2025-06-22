module bspline_evaluator #(
    parameter DATA_WIDTH = 32,
    parameter DEGREE = 3,
    parameter NUM_KNOTS = 16,
    parameter PIPELINE_DEPTH = 4
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] x,
    input wire [DATA_WIDTH-1:0] knots [NUM_KNOTS-1:0],
    input wire x_valid,
    
    output wire [DATA_WIDTH-1:0] basis_value,
    output wire basis_valid,
    output wire computation_done
);

reg [DATA_WIDTH-1:0] basis_functions [DEGREE+1:0][NUM_KNOTS-1:0];
reg [DATA_WIDTH-1:0] temp_basis [NUM_KNOTS-1:0];
reg [3:0] degree_counter;
reg [7:0] knot_index;
reg [2:0] computation_state;

wire [DATA_WIDTH-1:0] normalized_x;
wire [DATA_WIDTH-1:0] knot_span;
reg [DATA_WIDTH-1:0] alpha;

localparam IDLE = 3'h0;
localparam FIND_SPAN = 3'h1;
localparam COMPUTE_BASIS = 3'h2;
localparam UPDATE_BASIS = 3'h3;
localparam OUTPUT = 3'h4;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        degree_counter <= 4'h0;
        knot_index <= 8'h0;
        computation_state <= IDLE;
        alpha <= '0;
        for (int i = 0; i <= DEGREE; i = i + 1) begin
            for (int j = 0; j < NUM_KNOTS; j = j + 1) begin
                basis_functions[i][j] <= '0;
            end
        end
        for (int k = 0; k < NUM_KNOTS; k = k + 1) begin
            temp_basis[k] <= '0;
        end
    end else if (enable && x_valid) begin
        case (computation_state)
            IDLE: begin
                computation_state <= FIND_SPAN;
                knot_index <= 8'h0;
                degree_counter <= 4'h0;
                basis_functions[0][0] <= 32'h1;
            end
            
            FIND_SPAN: begin
                for (int m = 0; m < NUM_KNOTS-1; m = m + 1) begin
                    if (x >= knots[m] && x < knots[m+1]) begin
                        knot_index <= m;
                        break;
                    end
                end
                computation_state <= COMPUTE_BASIS;
            end
            
            COMPUTE_BASIS: begin
                if (degree_counter < DEGREE) begin
                    for (int n = 0; n <= degree_counter; n = n + 1) begin
                        if (knot_index >= n && knot_index + degree_counter + 1 - n < NUM_KNOTS) begin
                            if (knots[knot_index + degree_counter + 1 - n] != knots[knot_index - n]) begin
                                alpha <= (x - knots[knot_index - n]) / 
                                        (knots[knot_index + degree_counter + 1 - n] - knots[knot_index - n]);
                            end else begin
                                alpha <= '0;
                            end
                            
                            temp_basis[n] <= alpha * basis_functions[degree_counter][n] +
                                            (32'h1 - alpha) * basis_functions[degree_counter][n+1];
                        end
                    end
                    computation_state <= UPDATE_BASIS;
                end else begin
                    computation_state <= OUTPUT;
                end
            end
            
            UPDATE_BASIS: begin
                for (int p = 0; p <= degree_counter; p = p + 1) begin
                    basis_functions[degree_counter + 1][p] <= temp_basis[p];
                end
                degree_counter <= degree_counter + 1;
                computation_state <= COMPUTE_BASIS;
            end
            
            OUTPUT: begin
                computation_state <= IDLE;
            end
        endcase
    end
end

assign basis_value = basis_functions[DEGREE][0];
assign basis_valid = (computation_state == OUTPUT);
assign computation_done = (computation_state == OUTPUT);

endmodule

module bspline_coefficient_updater #(
    parameter DATA_WIDTH = 32,
    parameter NUM_COEFFS = 64,
    parameter ADDR_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [ADDR_WIDTH-1:0] addr,
    input wire [DATA_WIDTH-1:0] new_coeff,
    input wire update_en,
    
    output reg [DATA_WIDTH-1:0] coefficients [NUM_COEFFS-1:0],
    output wire update_done
);

reg update_active;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        update_active <= 1'b0;
        for (int i = 0; i < NUM_COEFFS; i = i + 1) begin
            coefficients[i] <= 32'h0;
        end
    end else if (enable) begin
        if (update_en && addr < NUM_COEFFS) begin
            coefficients[addr] <= new_coeff;
            update_active <= 1'b1;
        end else begin
            update_active <= 1'b0;
        end
    end
end

assign update_done = !update_active;

endmodule

module bspline_derivative_calculator #(
    parameter DATA_WIDTH = 32,
    parameter DEGREE = 3,
    parameter NUM_KNOTS = 16
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] x,
    input wire [DATA_WIDTH-1:0] knots [NUM_KNOTS-1:0],
    input wire [DATA_WIDTH-1:0] coefficients [NUM_KNOTS-DEGREE-1:0],
    input wire x_valid,
    
    output wire [DATA_WIDTH-1:0] derivative_value,
    output wire derivative_valid
);

wire [DATA_WIDTH-1:0] basis_left, basis_right;
wire [DATA_WIDTH-1:0] knot_diff_left, knot_diff_right;
reg [DATA_WIDTH-1:0] derivative_calc;
reg derivative_ready;

bspline_evaluator #(
    .DATA_WIDTH(DATA_WIDTH),
    .DEGREE(DEGREE-1),
    .NUM_KNOTS(NUM_KNOTS)
) left_basis (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .x(x),
    .knots(knots),
    .x_valid(x_valid),
    .basis_value(basis_left),
    .basis_valid(),
    .computation_done()
);

bspline_evaluator #(
    .DATA_WIDTH(DATA_WIDTH),
    .DEGREE(DEGREE-1),
    .NUM_KNOTS(NUM_KNOTS)
) right_basis (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .x(x),
    .knots(knots[1:NUM_KNOTS-1]),
    .x_valid(x_valid),
    .basis_value(basis_right),
    .basis_valid(),
    .computation_done()
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        derivative_calc <= '0;
        derivative_ready <= 1'b0;
    end else if (enable) begin
        knot_diff_left = knots[DEGREE] - knots[0];
        knot_diff_right = knots[DEGREE+1] - knots[1];
        
        if (knot_diff_left != 0 && knot_diff_right != 0) begin
            derivative_calc <= DEGREE * (basis_left / knot_diff_left - basis_right / knot_diff_right);
        end else begin
            derivative_calc <= '0;
        end
        derivative_ready <= x_valid;
    end
end

assign derivative_value = derivative_calc;
assign derivative_valid = derivative_ready;

endmodule

module bspline_knot_insertion #(
    parameter DATA_WIDTH = 32,
    parameter NUM_KNOTS = 16,
    parameter MAX_INSERTIONS = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] knot_vector_in [NUM_KNOTS-1:0],
    input wire [DATA_WIDTH-1:0] new_knot,
    input wire insert_en,
    
    output reg [DATA_WIDTH-1:0] knot_vector_out [NUM_KNOTS+MAX_INSERTIONS-1:0],
    output reg [7:0] new_knot_count,
    output wire insertion_done
);

reg [7:0] insertion_index;
reg [2:0] state;
reg [7:0] shift_index;

localparam IDLE = 3'h0;
localparam FIND_POSITION = 3'h1;
localparam SHIFT_KNOTS = 3'h2;
localparam INSERT_KNOT = 3'h3;
localparam DONE = 3'h4;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        insertion_index <= 8'h0;
        state <= IDLE;
        shift_index <= 8'h0;
        new_knot_count <= NUM_KNOTS;
        for (int i = 0; i < NUM_KNOTS + MAX_INSERTIONS; i = i + 1) begin
            if (i < NUM_KNOTS) begin
                knot_vector_out[i] <= knot_vector_in[i];
            end else begin
                knot_vector_out[i] <= '0;
            end
        end
    end else if (enable && insert_en) begin
        case (state)
            IDLE: begin
                state <= FIND_POSITION;
                insertion_index <= 8'h0;
            end
            
            FIND_POSITION: begin
                for (int j = 0; j < new_knot_count; j = j + 1) begin
                    if (new_knot <= knot_vector_out[j]) begin
                        insertion_index <= j;
                        break;
                    end
                end
                state <= SHIFT_KNOTS;
                shift_index <= new_knot_count;
            end
            
            SHIFT_KNOTS: begin
                if (shift_index > insertion_index) begin
                    knot_vector_out[shift_index] <= knot_vector_out[shift_index - 1];
                    shift_index <= shift_index - 1;
                end else begin
                    state <= INSERT_KNOT;
                end
            end
            
            INSERT_KNOT: begin
                knot_vector_out[insertion_index] <= new_knot;
                new_knot_count <= new_knot_count + 1;
                state <= DONE;
            end
            
            DONE: begin
                state <= IDLE;
            end
        endcase
    end
end

assign insertion_done = (state == DONE);

endmodule

module bspline_surface_evaluator #(
    parameter DATA_WIDTH = 32,
    parameter U_KNOTS = 8,
    parameter V_KNOTS = 8,
    parameter U_DEGREE = 3,
    parameter V_DEGREE = 3
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] u,
    input wire [DATA_WIDTH-1:0] v,
    input wire [DATA_WIDTH-1:0] u_knots [U_KNOTS-1:0],
    input wire [DATA_WIDTH-1:0] v_knots [V_KNOTS-1:0],
    input wire [DATA_WIDTH-1:0] control_points [U_KNOTS-U_DEGREE-1:0][V_KNOTS-V_DEGREE-1:0],
    input wire uv_valid,
    
    output wire [DATA_WIDTH-1:0] surface_point,
    output wire point_valid
);

wire [DATA_WIDTH-1:0] u_basis_values [U_KNOTS-U_DEGREE-1:0];
wire [DATA_WIDTH-1:0] v_basis_values [V_KNOTS-V_DEGREE-1:0];
wire u_basis_valid, v_basis_valid;

reg [DATA_WIDTH-1:0] surface_calc;
reg surface_ready;

bspline_evaluator #(
    .DATA_WIDTH(DATA_WIDTH),
    .DEGREE(U_DEGREE),
    .NUM_KNOTS(U_KNOTS)
) u_evaluator (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .x(u),
    .knots(u_knots),
    .x_valid(uv_valid),
    .basis_value(u_basis_values[0]),
    .basis_valid(u_basis_valid),
    .computation_done()
);

bspline_evaluator #(
    .DATA_WIDTH(DATA_WIDTH),
    .DEGREE(V_DEGREE),
    .NUM_KNOTS(V_KNOTS)
) v_evaluator (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .x(v),
    .knots(v_knots),
    .x_valid(uv_valid),
    .basis_value(v_basis_values[0]),
    .basis_valid(v_basis_valid),
    .computation_done()
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        surface_calc <= '0;
        surface_ready <= 1'b0;
    end else if (enable && u_basis_valid && v_basis_valid) begin
        surface_calc = '0;
        for (int i = 0; i < U_KNOTS-U_DEGREE-1; i = i + 1) begin
            for (int j = 0; j < V_KNOTS-V_DEGREE-1; j = j + 1) begin
                surface_calc = surface_calc + 
                              control_points[i][j] * u_basis_values[i] * v_basis_values[j];
            end
        end
        surface_ready <= 1'b1;
    end else begin
        surface_ready <= 1'b0;
    end
end

assign surface_point = surface_calc;
assign point_valid = surface_ready;

endmodule 