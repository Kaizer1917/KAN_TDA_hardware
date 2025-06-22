module parallel_reduction_engine #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256,
    parameter NUM_PE = 16,
    parameter PIPELINE_DEPTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] boundary_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0],
    input wire matrix_valid,
    
    output wire [DATA_WIDTH-1:0] reduced_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0],
    output wire reduction_valid,
    output wire [7:0] progress_percent,
    output wire computation_done
);

reg [DATA_WIDTH-1:0] working_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
reg [DATA_WIDTH-1:0] temp_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];

reg [7:0] current_column;
reg [7:0] pivot_row [NUM_PE-1:0];
reg [NUM_PE-1:0] pe_active;
reg [NUM_PE-1:0] pe_done;

wire [DATA_WIDTH-1:0] pe_outputs [NUM_PE-1:0][MATRIX_SIZE-1:0];
wire [NUM_PE-1:0] pe_valid;

reg [3:0] reduction_state;
reg [15:0] operation_counter;

localparam IDLE = 4'h0;
localparam COPY_MATRIX = 4'h1;
localparam FIND_PIVOTS = 4'h2;
localparam PARALLEL_REDUCE = 4'h3;
localparam MERGE_RESULTS = 4'h4;
localparam NEXT_COLUMN = 4'h5;
localparam OUTPUT = 4'h6;
localparam DONE = 4'h7;

genvar i;
generate
    for (i = 0; i < NUM_PE; i = i + 1) begin : reduction_units
        column_reduction_pe #(
            .DATA_WIDTH(DATA_WIDTH),
            .MATRIX_SIZE(MATRIX_SIZE),
            .PE_ID(i)
        ) pe (
            .clk(clk),
            .rst_n(rst_n),
            .enable(pe_active[i]),
            .matrix_column(working_matrix[current_column + i]),
            .pivot_row(pivot_row[i]),
            .reference_columns(working_matrix),
            .reduced_column(pe_outputs[i]),
            .reduction_done(pe_done[i]),
            .valid_out(pe_valid[i])
        );
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_column <= 8'h0;
        reduction_state <= IDLE;
        pe_active <= '0;
        operation_counter <= 16'h0;
        for (int j = 0; j < MATRIX_SIZE; j = j + 1) begin
            for (int k = 0; k < MATRIX_SIZE; k = k + 1) begin
                working_matrix[j][k] <= '0;
                temp_matrix[j][k] <= '0;
            end
        end
        for (int m = 0; m < NUM_PE; m = m + 1) begin
            pivot_row[m] <= 8'hFF;
        end
    end else if (enable) begin
        case (reduction_state)
            IDLE: begin
                if (matrix_valid) begin
                    reduction_state <= COPY_MATRIX;
                    operation_counter <= 16'h0;
                end
            end
            
            COPY_MATRIX: begin
                for (int n = 0; n < MATRIX_SIZE; n = n + 1) begin
                    for (int p = 0; p < MATRIX_SIZE; p = p + 1) begin
                        working_matrix[n][p] <= boundary_matrix[n][p];
                    end
                end
                reduction_state <= FIND_PIVOTS;
                current_column <= 8'h0;
            end
            
            FIND_PIVOTS: begin
                for (int q = 0; q < NUM_PE; q = q + 1) begin
                    if (current_column + q < MATRIX_SIZE) begin
                        pivot_row[q] <= find_pivot_row(current_column + q);
                    end else begin
                        pivot_row[q] <= 8'hFF;
                    end
                end
                reduction_state <= PARALLEL_REDUCE;
                pe_active <= {NUM_PE{1'b1}};
                operation_counter <= operation_counter + 1;
            end
            
            PARALLEL_REDUCE: begin
                if (&pe_done) begin
                    reduction_state <= MERGE_RESULTS;
                    pe_active <= '0;
                end
            end
            
            MERGE_RESULTS: begin
                for (int r = 0; r < NUM_PE; r = r + 1) begin
                    if (current_column + r < MATRIX_SIZE && pe_valid[r]) begin
                        for (int s = 0; s < MATRIX_SIZE; s = s + 1) begin
                            working_matrix[s][current_column + r] <= pe_outputs[r][s];
                        end
                    end
                end
                reduction_state <= NEXT_COLUMN;
            end
            
            NEXT_COLUMN: begin
                current_column <= current_column + NUM_PE;
                if (current_column + NUM_PE >= MATRIX_SIZE) begin
                    reduction_state <= OUTPUT;
                end else begin
                    reduction_state <= FIND_PIVOTS;
                end
            end
            
            OUTPUT: begin
                for (int t = 0; t < MATRIX_SIZE; t = t + 1) begin
                    for (int u = 0; u < MATRIX_SIZE; u = u + 1) begin
                        temp_matrix[t][u] <= working_matrix[t][u];
                    end
                end
                reduction_state <= DONE;
            end
            
            DONE: begin
                reduction_state <= IDLE;
            end
        endcase
    end
end

function [7:0] find_pivot_row;
    input [7:0] column;
    reg [7:0] row;
    begin
        find_pivot_row = 8'hFF;
        for (row = MATRIX_SIZE - 1; row >= 0; row = row - 1) begin
            if (working_matrix[row][column] != 0) begin
                find_pivot_row = row;
                break;
            end
        end
    end
endfunction

assign reduced_matrix = temp_matrix;
assign reduction_valid = (reduction_state == DONE);
assign progress_percent = (current_column * 100) / MATRIX_SIZE;
assign computation_done = (reduction_state == DONE);

endmodule

module column_reduction_pe #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256,
    parameter PE_ID = 0
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] matrix_column [MATRIX_SIZE-1:0],
    input wire [7:0] pivot_row,
    input wire [DATA_WIDTH-1:0] reference_columns [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0],
    
    output reg [DATA_WIDTH-1:0] reduced_column [MATRIX_SIZE-1:0],
    output reg reduction_done,
    output reg valid_out
);

reg [DATA_WIDTH-1:0] working_column [MATRIX_SIZE-1:0];
reg [7:0] ref_col_index;
reg [2:0] pe_state;
reg [7:0] row_counter;

localparam PE_IDLE = 3'h0;
localparam COPY_COLUMN = 3'h1;
localparam FIND_REDUCER = 3'h2;
localparam REDUCE_COLUMN = 3'h3;
localparam CHECK_COMPLETE = 3'h4;
localparam PE_DONE = 3'h5;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pe_state <= PE_IDLE;
        ref_col_index <= 8'h0;
        row_counter <= 8'h0;
        reduction_done <= 1'b0;
        valid_out <= 1'b0;
        for (int i = 0; i < MATRIX_SIZE; i = i + 1) begin
            working_column[i] <= '0;
            reduced_column[i] <= '0;
        end
    end else if (enable) begin
        case (pe_state)
            PE_IDLE: begin
                pe_state <= COPY_COLUMN;
                ref_col_index <= 8'h0;
                reduction_done <= 1'b0;
                valid_out <= 1'b0;
            end
            
            COPY_COLUMN: begin
                for (int j = 0; j < MATRIX_SIZE; j = j + 1) begin
                    working_column[j] <= matrix_column[j];
                end
                pe_state <= FIND_REDUCER;
            end
            
            FIND_REDUCER: begin
                if (pivot_row != 8'hFF) begin
                    for (int k = 0; k < ref_col_index; k = k + 1) begin
                        if (reference_columns[pivot_row][k] != 0 && 
                            reference_columns[pivot_row][k] == working_column[pivot_row]) begin
                            pe_state <= REDUCE_COLUMN;
                            row_counter <= 8'h0;
                            break;
                        end
                    end
                    if (pe_state != REDUCE_COLUMN) begin
                        pe_state <= CHECK_COMPLETE;
                    end
                end else begin
                    pe_state <= CHECK_COMPLETE;
                end
            end
            
            REDUCE_COLUMN: begin
                if (row_counter < MATRIX_SIZE) begin
                    working_column[row_counter] <= working_column[row_counter] ^ 
                                                  reference_columns[row_counter][ref_col_index];
                    row_counter <= row_counter + 1;
                end else begin
                    pe_state <= FIND_REDUCER;
                    ref_col_index <= ref_col_index + 1;
                end
            end
            
            CHECK_COMPLETE: begin
                for (int m = 0; m < MATRIX_SIZE; m = m + 1) begin
                    reduced_column[m] <= working_column[m];
                end
                pe_state <= PE_DONE;
                valid_out <= 1'b1;
            end
            
            PE_DONE: begin
                reduction_done <= 1'b1;
                pe_state <= PE_IDLE;
            end
        endcase
    end
end

endmodule

module persistence_pair_extractor #(
    parameter DATA_WIDTH = 32,
    parameter MATRIX_SIZE = 256,
    parameter MAX_PAIRS = 128
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] reduced_matrix [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0],
    input wire matrix_valid,
    
    output reg [DATA_WIDTH-1:0] birth_times [MAX_PAIRS-1:0],
    output reg [DATA_WIDTH-1:0] death_times [MAX_PAIRS-1:0],
    output reg [7:0] num_pairs,
    output reg extraction_done
);

reg [7:0] current_col, current_row;
reg [2:0] extract_state;
reg [7:0] pair_counter;
reg [7:0] pivot_positions [MATRIX_SIZE-1:0];

localparam EXTRACT_IDLE = 3'h0;
localparam SCAN_MATRIX = 3'h1;
localparam FIND_PIVOTS = 3'h2;
localparam EXTRACT_PAIRS = 3'h3;
localparam EXTRACT_DONE = 3'h4;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_col <= 8'h0;
        current_row <= 8'h0;
        extract_state <= EXTRACT_IDLE;
        pair_counter <= 8'h0;
        num_pairs <= 8'h0;
        extraction_done <= 1'b0;
        for (int i = 0; i < MAX_PAIRS; i = i + 1) begin
            birth_times[i] <= '0;
            death_times[i] <= '0;
        end
        for (int j = 0; j < MATRIX_SIZE; j = j + 1) begin
            pivot_positions[j] <= 8'hFF;
        end
    end else if (enable && matrix_valid) begin
        case (extract_state)
            EXTRACT_IDLE: begin
                extract_state <= SCAN_MATRIX;
                current_col <= 8'h0;
                current_row <= 8'h0;
                pair_counter <= 8'h0;
            end
            
            SCAN_MATRIX: begin
                extract_state <= FIND_PIVOTS;
                current_col <= 8'h0;
            end
            
            FIND_PIVOTS: begin
                for (int k = MATRIX_SIZE - 1; k >= 0; k = k - 1) begin
                    if (reduced_matrix[k][current_col] != 0) begin
                        pivot_positions[current_col] <= k;
                        break;
                    end
                end
                
                current_col <= current_col + 1;
                if (current_col >= MATRIX_SIZE - 1) begin
                    extract_state <= EXTRACT_PAIRS;
                    current_col <= 8'h0;
                end
            end
            
            EXTRACT_PAIRS: begin
                if (current_col < MATRIX_SIZE && pair_counter < MAX_PAIRS) begin
                    if (pivot_positions[current_col] != 8'hFF) begin
                        birth_times[pair_counter] <= {24'h0, pivot_positions[current_col]};
                        death_times[pair_counter] <= {24'h0, current_col};
                        pair_counter <= pair_counter + 1;
                    end
                end
                
                current_col <= current_col + 1;
                if (current_col >= MATRIX_SIZE - 1) begin
                    extract_state <= EXTRACT_DONE;
                    num_pairs <= pair_counter;
                end
            end
            
            EXTRACT_DONE: begin
                extraction_done <= 1'b1;
                extract_state <= EXTRACT_IDLE;
            end
        endcase
    end
end

endmodule

module persistence_diagram_generator #(
    parameter DATA_WIDTH = 32,
    parameter MAX_PAIRS = 128,
    parameter COORD_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] birth_times [MAX_PAIRS-1:0],
    input wire [DATA_WIDTH-1:0] death_times [MAX_PAIRS-1:0],
    input wire [7:0] num_pairs,
    input wire pairs_valid,
    
    output reg [COORD_WIDTH-1:0] diagram_x [MAX_PAIRS-1:0],
    output reg [COORD_WIDTH-1:0] diagram_y [MAX_PAIRS-1:0],
    output reg [7:0] valid_points,
    output reg diagram_ready
);

reg [7:0] point_index;
reg [1:0] diagram_state;
reg [DATA_WIDTH-1:0] max_death_time;

localparam DIAGRAM_IDLE = 2'h0;
localparam NORMALIZE_COORDS = 2'h1;
localparam DIAGRAM_DONE = 2'h2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        point_index <= 8'h0;
        diagram_state <= DIAGRAM_IDLE;
        max_death_time <= '0;
        valid_points <= 8'h0;
        diagram_ready <= 1'b0;
        for (int i = 0; i < MAX_PAIRS; i = i + 1) begin
            diagram_x[i] <= '0;
            diagram_y[i] <= '0;
        end
    end else if (enable && pairs_valid) begin
        case (diagram_state)
            DIAGRAM_IDLE: begin
                diagram_state <= NORMALIZE_COORDS;
                point_index <= 8'h0;
                max_death_time <= '0;
                
                for (int j = 0; j < num_pairs; j = j + 1) begin
                    if (death_times[j] > max_death_time) begin
                        max_death_time <= death_times[j];
                    end
                end
            end
            
            NORMALIZE_COORDS: begin
                if (point_index < num_pairs) begin
                    if (max_death_time != 0) begin
                        diagram_x[point_index] <= (birth_times[point_index] * {COORD_WIDTH{1'b1}}) / max_death_time;
                        diagram_y[point_index] <= (death_times[point_index] * {COORD_WIDTH{1'b1}}) / max_death_time;
                    end else begin
                        diagram_x[point_index] <= '0;
                        diagram_y[point_index] <= '0;
                    end
                    point_index <= point_index + 1;
                end else begin
                    diagram_state <= DIAGRAM_DONE;
                    valid_points <= point_index;
                end
            end
            
            DIAGRAM_DONE: begin
                diagram_ready <= 1'b1;
                diagram_state <= DIAGRAM_IDLE;
            end
        endcase
    end
end

endmodule

module persistence_landscape_computer #(
    parameter DATA_WIDTH = 32,
    parameter MAX_PAIRS = 128,
    parameter RESOLUTION = 256,
    parameter NUM_LANDSCAPES = 5
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    input wire [DATA_WIDTH-1:0] birth_times [MAX_PAIRS-1:0],
    input wire [DATA_WIDTH-1:0] death_times [MAX_PAIRS-1:0],
    input wire [7:0] num_pairs,
    input wire pairs_valid,
    
    output reg [DATA_WIDTH-1:0] landscape_values [NUM_LANDSCAPES-1:0][RESOLUTION-1:0],
    output reg landscape_ready
);

reg [7:0] current_pair;
reg [7:0] current_point;
reg [7:0] current_landscape;
reg [2:0] landscape_state;
reg [DATA_WIDTH-1:0] t_values [RESOLUTION-1:0];
reg [DATA_WIDTH-1:0] temp_landscapes [MAX_PAIRS-1:0][RESOLUTION-1:0];

localparam LANDSCAPE_IDLE = 3'h0;
localparam INIT_T_VALUES = 3'h1;
localparam COMPUTE_FUNCTIONS = 3'h2;
localparam SORT_LANDSCAPES = 3'h3;
localparam LANDSCAPE_DONE = 3'h4;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_pair <= 8'h0;
        current_point <= 8'h0;
        current_landscape <= 8'h0;
        landscape_state <= LANDSCAPE_IDLE;
        landscape_ready <= 1'b0;
        
        for (int i = 0; i < NUM_LANDSCAPES; i = i + 1) begin
            for (int j = 0; j < RESOLUTION; j = j + 1) begin
                landscape_values[i][j] <= '0;
            end
        end
        
        for (int k = 0; k < MAX_PAIRS; k = k + 1) begin
            for (int l = 0; l < RESOLUTION; l = l + 1) begin
                temp_landscapes[k][l] <= '0;
            end
        end
        
        for (int m = 0; m < RESOLUTION; m = m + 1) begin
            t_values[m] <= '0;
        end
    end else if (enable && pairs_valid) begin
        case (landscape_state)
            LANDSCAPE_IDLE: begin
                landscape_state <= INIT_T_VALUES;
                current_point <= 8'h0;
            end
            
            INIT_T_VALUES: begin
                if (current_point < RESOLUTION) begin
                    t_values[current_point] <= (current_point * 32'h1000) / RESOLUTION;
                    current_point <= current_point + 1;
                end else begin
                    landscape_state <= COMPUTE_FUNCTIONS;
                    current_pair <= 8'h0;
                    current_point <= 8'h0;
                end
            end
            
            COMPUTE_FUNCTIONS: begin
                if (current_pair < num_pairs && current_point < RESOLUTION) begin
                    if (t_values[current_point] >= birth_times[current_pair] && 
                        t_values[current_point] <= death_times[current_pair]) begin
                        
                        if (t_values[current_point] <= (birth_times[current_pair] + death_times[current_pair]) / 2) begin
                            temp_landscapes[current_pair][current_point] <= 
                                t_values[current_point] - birth_times[current_pair];
                        end else begin
                            temp_landscapes[current_pair][current_point] <= 
                                death_times[current_pair] - t_values[current_point];
                        end
                    end else begin
                        temp_landscapes[current_pair][current_point] <= '0;
                    end
                    
                    current_point <= current_point + 1;
                    if (current_point >= RESOLUTION - 1) begin
                        current_point <= 8'h0;
                        current_pair <= current_pair + 1;
                        if (current_pair >= num_pairs - 1) begin
                            landscape_state <= SORT_LANDSCAPES;
                            current_point <= 8'h0;
                            current_landscape <= 8'h0;
                        end
                    end
                end
            end
            
            SORT_LANDSCAPES: begin
                if (current_point < RESOLUTION && current_landscape < NUM_LANDSCAPES) begin
                    landscape_values[current_landscape][current_point] <= 
                        find_kth_largest(current_point, current_landscape);
                    
                    current_landscape <= current_landscape + 1;
                    if (current_landscape >= NUM_LANDSCAPES - 1) begin
                        current_landscape <= 8'h0;
                        current_point <= current_point + 1;
                        if (current_point >= RESOLUTION - 1) begin
                            landscape_state <= LANDSCAPE_DONE;
                        end
                    end
                end
            end
            
            LANDSCAPE_DONE: begin
                landscape_ready <= 1'b1;
                landscape_state <= LANDSCAPE_IDLE;
            end
        endcase
    end
end

function [DATA_WIDTH-1:0] find_kth_largest;
    input [7:0] point_idx;
    input [7:0] k;
    reg [DATA_WIDTH-1:0] sorted_values [MAX_PAIRS-1:0];
    reg [7:0] i, j;
    reg [DATA_WIDTH-1:0] temp_val;
    begin
        for (i = 0; i < num_pairs; i = i + 1) begin
            sorted_values[i] = temp_landscapes[i][point_idx];
        end
        
        for (i = 0; i < num_pairs - 1; i = i + 1) begin
            for (j = 0; j < num_pairs - 1 - i; j = j + 1) begin
                if (sorted_values[j] < sorted_values[j + 1]) begin
                    temp_val = sorted_values[j];
                    sorted_values[j] = sorted_values[j + 1];
                    sorted_values[j + 1] = temp_val;
                end
            end
        end
        
        if (k < num_pairs) begin
            find_kth_largest = sorted_values[k];
        end else begin
            find_kth_largest = '0;
        end
    end
endfunction

endmodule 