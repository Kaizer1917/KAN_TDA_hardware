module tda_acceleration_unit #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256,
    parameter NUM_PE = 16,
    parameter FIFO_DEPTH = 512,
    parameter MAX_DIMENSION = 3
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] point_data,
    input wire point_valid,
    output wire point_ready,
    
    output wire [DATA_WIDTH-1:0] persistence_data,
    output wire persistence_valid,
    input wire persistence_ready,
    
    input wire [7:0] num_points,
    input wire [7:0] dimension,
    input wire [DATA_WIDTH-1:0] epsilon,
    
    output wire computation_active,
    output wire [7:0] progress_counter,
    output wire error_flag
);

wire [DATA_WIDTH-1:0] distance_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
wire distance_matrix_valid;
wire [DATA_WIDTH-1:0] boundary_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
wire boundary_matrix_valid;

wire [DATA_WIDTH-1:0] point_buffer [MATRIX_SIZE-1:0][MAX_DIMENSION-1:0];
wire [7:0] point_counter;
wire points_loaded;

distance_computation_engine #(
    .DATA_WIDTH(DATA_WIDTH),
    .MATRIX_SIZE(MATRIX_SIZE),
    .MAX_DIMENSION(MAX_DIMENSION)
) distance_engine (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .points(point_buffer),
    .num_points(num_points),
    .dimension(dimension),
    .distance_matrix(distance_matrix),
    .matrix_valid(distance_matrix_valid),
    .computation_done()
);

simplicial_complex_builder #(
    .DATA_WIDTH(DATA_WIDTH),
    .MATRIX_SIZE(MATRIX_SIZE),
    .NUM_PE(NUM_PE)
) complex_builder (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable && distance_matrix_valid),
    .distance_matrix(distance_matrix),
    .epsilon(epsilon),
    .boundary_matrix(boundary_matrix),
    .boundary_valid(boundary_matrix_valid),
    .progress(progress_counter)
);

persistence_homology_engine #(
    .DATA_WIDTH(DATA_WIDTH),
    .MATRIX_SIZE(MATRIX_SIZE)
) persistence_engine (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable && boundary_matrix_valid),
    .boundary_matrix(boundary_matrix),
    .persistence_pairs(persistence_data),
    .persistence_valid(persistence_valid),
    .computation_done(computation_active)
);

point_buffer_controller #(
    .DATA_WIDTH(DATA_WIDTH),
    .MATRIX_SIZE(MATRIX_SIZE),
    .MAX_DIMENSION(MAX_DIMENSION)
) buffer_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .point_data(point_data),
    .point_valid(point_valid),
    .point_ready(point_ready),
    .dimension(dimension),
    .point_buffer(point_buffer),
    .point_counter(point_counter),
    .points_loaded(points_loaded)
);

assign error_flag = (point_counter > num_points) || 
                   (dimension > MAX_DIMENSION) || 
                   (num_points > MATRIX_SIZE);

endmodule

module distance_computation_engine #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256,
    parameter MAX_DIMENSION = 3
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] points [MATRIX_SIZE-1:0][MAX_DIMENSION-1:0],
    input wire [7:0] num_points,
    input wire [7:0] dimension,
    
    output reg [DATA_WIDTH-1:0] distance_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0],
    output reg matrix_valid,
    output reg computation_done
);

reg [7:0] i_counter, j_counter;
reg [3:0] dim_counter;
reg [DATA_WIDTH-1:0] temp_diff;
reg [DATA_WIDTH-1:0] sum_squares;
reg [2:0] state;

localparam IDLE = 3'h0;
localparam COMPUTE_DIFF = 3'h1;
localparam ACCUMULATE = 3'h2;
localparam SQRT_COMPUTE = 3'h3;
localparam STORE_RESULT = 3'h4;
localparam DONE = 3'h5;

sqrt_unit #(
    .DATA_WIDTH(DATA_WIDTH)
) sqrt_calc (
    .clk(clk),
    .rst_n(rst_n),
    .data_in(sum_squares),
    .data_out(),
    .valid_out()
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        i_counter <= 8'h0;
        j_counter <= 8'h0;
        dim_counter <= 3'h0;
        sum_squares <= '0;
        temp_diff <= '0;
        state <= IDLE;
        matrix_valid <= 1'b0;
        computation_done <= 1'b0;
        for (int x = 0; x < MATRIX_SIZE; x = x + 1) begin
            for (int y = 0; y < MATRIX_SIZE; y = y + 1) begin
                distance_matrix[x][y] <= '0;
            end
        end
    end else if (enable) begin
        case (state)
            IDLE: begin
                if (num_points > 0) begin
                    state <= COMPUTE_DIFF;
                    i_counter <= 8'h0;
                    j_counter <= 8'h0;
                    dim_counter <= 3'h0;
                    sum_squares <= '0;
                end
            end
            
            COMPUTE_DIFF: begin
                if (dim_counter < dimension) begin
                    temp_diff <= points[i_counter][dim_counter] > points[j_counter][dim_counter] ?
                                points[i_counter][dim_counter] - points[j_counter][dim_counter] :
                                points[j_counter][dim_counter] - points[i_counter][dim_counter];
                    state <= ACCUMULATE;
                end else begin
                    state <= SQRT_COMPUTE;
                    dim_counter <= 3'h0;
                end
            end
            
            ACCUMULATE: begin
                sum_squares <= sum_squares + (temp_diff * temp_diff);
                dim_counter <= dim_counter + 1;
                state <= COMPUTE_DIFF;
            end
            
            SQRT_COMPUTE: begin
                state <= STORE_RESULT;
            end
            
            STORE_RESULT: begin
                distance_matrix[i_counter][j_counter] <= sum_squares[DATA_WIDTH-1:DATA_WIDTH/2];
                distance_matrix[j_counter][i_counter] <= sum_squares[DATA_WIDTH-1:DATA_WIDTH/2];
                
                j_counter <= j_counter + 1;
                sum_squares <= '0;
                
                if (j_counter >= num_points - 1) begin
                    j_counter <= 8'h0;
                    i_counter <= i_counter + 1;
                    
                    if (i_counter >= num_points - 1) begin
                        state <= DONE;
                        matrix_valid <= 1'b1;
                        computation_done <= 1'b1;
                    end else begin
                        state <= COMPUTE_DIFF;
                    end
                end else begin
                    state <= COMPUTE_DIFF;
                end
            end
            
            DONE: begin
                computation_done <= 1'b1;
            end
        endcase
    end
end

endmodule

module simplicial_complex_builder #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256,
    parameter NUM_PE = 16
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] distance_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0],
    input wire [DATA_WIDTH-1:0] epsilon,
    
    output reg [DATA_WIDTH-1:0] boundary_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0],
    output reg boundary_valid,
    output reg [7:0] progress
);

reg [7:0] current_row, current_col;
reg [3:0] state;
reg [NUM_PE-1:0] pe_active;
wire [NUM_PE-1:0] pe_done;

localparam IDLE = 4'h0;
localparam BUILD_EDGES = 4'h1;
localparam BUILD_TRIANGLES = 4'h2;
localparam COMPUTE_BOUNDARY = 4'h3;
localparam DONE = 4'h4;

genvar i;
generate
    for (i = 0; i < NUM_PE; i = i + 1) begin : processing_elements
        simplicial_pe #(
            .DATA_WIDTH(DATA_WIDTH),
            .MATRIX_SIZE(MATRIX_SIZE)
        ) pe (
            .clk(clk),
            .rst_n(rst_n),
            .enable(pe_active[i]),
            .distance_row(distance_matrix[current_row + i]),
            .epsilon(epsilon),
            .boundary_row(boundary_matrix[current_row + i]),
            .done(pe_done[i])
        );
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_row <= 8'h0;
        current_col <= 8'h0;
        state <= IDLE;
        pe_active <= '0;
        boundary_valid <= 1'b0;
        progress <= 8'h0;
        for (int j = 0; j < MATRIX_SIZE; j = j + 1) begin
            for (int k = 0; k < MATRIX_SIZE; k = k + 1) begin
                boundary_matrix[j][k] <= '0;
            end
        end
    end else if (enable) begin
        case (state)
            IDLE: begin
                state <= BUILD_EDGES;
                current_row <= 8'h0;
                pe_active <= {NUM_PE{1'b1}};
            end
            
            BUILD_EDGES: begin
                if (&pe_done) begin
                    current_row <= current_row + NUM_PE;
                    progress <= progress + NUM_PE;
                    
                    if (current_row + NUM_PE >= MATRIX_SIZE) begin
                        state <= BUILD_TRIANGLES;
                        current_row <= 8'h0;
                        pe_active <= '0;
                    end else begin
                        pe_active <= {NUM_PE{1'b1}};
                    end
                end
            end
            
            BUILD_TRIANGLES: begin
                state <= COMPUTE_BOUNDARY;
                current_row <= 8'h0;
            end
            
            COMPUTE_BOUNDARY: begin
                current_row <= current_row + 1;
                if (current_row >= MATRIX_SIZE - 1) begin
                    state <= DONE;
                    boundary_valid <= 1'b1;
                end
            end
            
            DONE: begin
                boundary_valid <= 1'b1;
            end
        endcase
    end
end

endmodule

module simplicial_pe #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] distance_row [MATRIX_SIZE-1:0],
    input wire [DATA_WIDTH-1:0] epsilon,
    
    output reg [DATA_WIDTH-1:0] boundary_row [MATRIX_SIZE-1:0],
    output reg done
);

reg [7:0] col_counter;
reg [1:0] state;

localparam IDLE = 2'h0;
localparam PROCESS = 2'h1;
localparam COMPLETE = 2'h2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        col_counter <= 8'h0;
        state <= IDLE;
        done <= 1'b0;
        for (int i = 0; i < MATRIX_SIZE; i = i + 1) begin
            boundary_row[i] <= '0;
        end
    end else if (enable) begin
        case (state)
            IDLE: begin
                state <= PROCESS;
                col_counter <= 8'h0;
                done <= 1'b0;
            end
            
            PROCESS: begin
                if (distance_row[col_counter] <= epsilon && distance_row[col_counter] != 0) begin
                    boundary_row[col_counter] <= 32'h1;
                end else begin
                    boundary_row[col_counter] <= 32'h0;
                end
                
                col_counter <= col_counter + 1;
                
                if (col_counter >= MATRIX_SIZE - 1) begin
                    state <= COMPLETE;
                end
            end
            
            COMPLETE: begin
                done <= 1'b1;
                state <= IDLE;
            end
        endcase
    end
end

endmodule

module persistence_homology_engine #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] boundary_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0],
    
    output reg [DATA_WIDTH-1:0] persistence_pairs,
    output reg persistence_valid,
    output reg computation_done
);

reg [DATA_WIDTH-1:0] reduced_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
reg [7:0] current_col;
reg [7:0] pivot_row;
reg [3:0] reduction_state;
reg [7:0] pairs_found;

localparam COPY_MATRIX = 4'h0;
localparam FIND_PIVOT = 4'h1;
localparam REDUCE_COLUMN = 4'h2;
localparam EXTRACT_PAIRS = 4'h3;
localparam OUTPUT_PAIRS = 4'h4;
localparam DONE = 4'h5;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_col <= 8'h0;
        pivot_row <= 8'h0;
        reduction_state <= COPY_MATRIX;
        persistence_pairs <= '0;
        persistence_valid <= 1'b0;
        computation_done <= 1'b0;
        pairs_found <= 8'h0;
        
        for (int i = 0; i < MATRIX_SIZE; i = i + 1) begin
            for (int j = 0; j < MATRIX_SIZE; j = j + 1) begin
                reduced_matrix[i][j] <= '0;
            end
        end
    end else if (enable) begin
        case (reduction_state)
            COPY_MATRIX: begin
                for (int k = 0; k < MATRIX_SIZE; k = k + 1) begin
                    for (int l = 0; l < MATRIX_SIZE; l = l + 1) begin
                        reduced_matrix[k][l] <= boundary_matrix[k][l];
                    end
                end
                reduction_state <= FIND_PIVOT;
                current_col <= 8'h0;
            end
            
            FIND_PIVOT: begin
                pivot_row <= 8'hFF;
                for (int m = MATRIX_SIZE-1; m >= 0; m = m - 1) begin
                    if (reduced_matrix[m][current_col] != 0) begin
                        pivot_row <= m;
                        break;
                    end
                end
                reduction_state <= REDUCE_COLUMN;
            end
            
            REDUCE_COLUMN: begin
                if (pivot_row != 8'hFF) begin
                    for (int n = 0; n < current_col; n = n + 1) begin
                        if (reduced_matrix[pivot_row][n] == reduced_matrix[pivot_row][current_col]) begin
                            for (int p = 0; p < MATRIX_SIZE; p = p + 1) begin
                                reduced_matrix[p][current_col] <= 
                                    reduced_matrix[p][current_col] ^ reduced_matrix[p][n];
                            end
                        end
                    end
                end
                
                current_col <= current_col + 1;
                if (current_col >= MATRIX_SIZE - 1) begin
                    reduction_state <= EXTRACT_PAIRS;
                    current_col <= 8'h0;
                end else begin
                    reduction_state <= FIND_PIVOT;
                end
            end
            
            EXTRACT_PAIRS: begin
                reduction_state <= OUTPUT_PAIRS;
                pairs_found <= 8'h0;
                current_col <= 8'h0;
            end
            
            OUTPUT_PAIRS: begin
                if (current_col < MATRIX_SIZE && reduced_matrix[0][current_col] != 0) begin
                    persistence_pairs <= {16'h0, current_col, 8'h0};
                    persistence_valid <= 1'b1;
                    pairs_found <= pairs_found + 1;
                end else begin
                    persistence_valid <= 1'b0;
                end
                
                current_col <= current_col + 1;
                
                if (current_col >= MATRIX_SIZE - 1) begin
                    reduction_state <= DONE;
                    computation_done <= 1'b1;
                end
            end
            
            DONE: begin
                computation_done <= 1'b1;
                persistence_valid <= 1'b0;
            end
        endcase
    end
end

endmodule

module point_buffer_controller #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256,
    parameter MAX_DIMENSION = 3
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] point_data,
    input wire point_valid,
    output reg point_ready,
    input wire [7:0] dimension,
    
    output reg [DATA_WIDTH-1:0] point_buffer [MATRIX_SIZE-1:0][MAX_DIMENSION-1:0],
    output reg [7:0] point_counter,
    output reg points_loaded
);

reg [2:0] dim_counter;
reg [1:0] state;

localparam IDLE = 2'h0;
localparam LOADING = 2'h1;
localparam DONE = 2'h2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        point_counter <= 8'h0;
        dim_counter <= 3'h0;
        state <= IDLE;
        point_ready <= 1'b1;
        points_loaded <= 1'b0;
        for (int i = 0; i < MATRIX_SIZE; i = i + 1) begin
            for (int j = 0; j < MAX_DIMENSION; j = j + 1) begin
                point_buffer[i][j] <= '0;
            end
        end
    end else if (enable) begin
        case (state)
            IDLE: begin
                if (point_valid) begin
                    state <= LOADING;
                    point_ready <= 1'b1;
                end
            end
            
            LOADING: begin
                if (point_valid && point_ready) begin
                    point_buffer[point_counter][dim_counter] <= point_data;
                    dim_counter <= dim_counter + 1;
                    
                    if (dim_counter >= dimension - 1) begin
                        dim_counter <= 3'h0;
                        point_counter <= point_counter + 1;
                        
                        if (point_counter >= MATRIX_SIZE - 1) begin
                            state <= DONE;
                            point_ready <= 1'b0;
                            points_loaded <= 1'b1;
                        end
                    end
                end
            end
            
            DONE: begin
                points_loaded <= 1'b1;
                point_ready <= 1'b0;
            end
        endcase
    end
end

endmodule

module sqrt_unit #(
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output reg [DATA_WIDTH-1:0] data_out,
    output reg valid_out
);

reg [DATA_WIDTH-1:0] x, y;
reg [3:0] iteration;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        x <= '0;
        y <= '0;
        iteration <= 4'h0;
        data_out <= '0;
        valid_out <= 1'b0;
    end else begin
        if (iteration == 0) begin
            x <= data_in;
            y <= data_in >> 1;
            iteration <= 4'h1;
            valid_out <= 1'b0;
        end else if (iteration < 8) begin
            y <= (y + x / y) >> 1;
            iteration <= iteration + 1;
        end else begin
            data_out <= y;
            valid_out <= 1'b1;
            iteration <= 4'h0;
        end
    end
end

endmodule 