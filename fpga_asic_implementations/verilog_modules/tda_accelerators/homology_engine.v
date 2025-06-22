module homology_engine #(
    parameter DATA_WIDTH = 16,
    parameter MAX_SIMPLICES = 4096,
    parameter MAX_DIMENSION = 3
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] simplex_data [0:MAX_SIMPLICES-1],
    input wire [11:0] num_simplices,
    input wire [1:0] max_dimension,
    input wire compute_start,
    output reg [DATA_WIDTH-1:0] betti_numbers [0:MAX_DIMENSION],
    output reg [DATA_WIDTH-1:0] persistence_pairs [0:1023][1:0],
    output reg [9:0] num_pairs,
    output reg computation_complete
);

    reg [DATA_WIDTH-1:0] boundary_matrix [0:MAX_SIMPLICES-1][0:MAX_SIMPLICES-1];
    reg [MAX_SIMPLICES-1:0] matrix_row [0:MAX_SIMPLICES-1];
    reg [11:0] matrix_size;
    reg [3:0] reduction_state;
    reg [11:0] current_col, current_row;
    reg [11:0] pivot_row [0:MAX_SIMPLICES-1];
    reg [MAX_SIMPLICES-1:0] reduced_matrix [0:MAX_SIMPLICES-1];
    reg [9:0] pair_count;
    
    function [11:0] find_pivot;
        input [MAX_SIMPLICES-1:0] column;
        input [11:0] size;
        reg [11:0] pivot;
        integer i;
        begin
            pivot = size;
            for (i = size - 1; i >= 0; i = i - 1) begin
                if (column[i]) begin
                    pivot = i;
                end
            end
            find_pivot = pivot;
        end
    endfunction
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < MAX_DIMENSION + 1; i = i + 1) begin
                betti_numbers[i] <= 0;
            end
            for (integer i = 0; i < 1024; i = i + 1) begin
                persistence_pairs[i][0] <= 0;
                persistence_pairs[i][1] <= 0;
            end
            for (integer i = 0; i < MAX_SIMPLICES; i = i + 1) begin
                pivot_row[i] <= MAX_SIMPLICES;
                for (integer j = 0; j < MAX_SIMPLICES; j = j + 1) begin
                    boundary_matrix[i][j] <= 0;
                end
                matrix_row[i] <= 0;
                reduced_matrix[i] <= 0;
            end
            reduction_state <= 0;
            current_col <= 0;
            current_row <= 0;
            matrix_size <= 0;
            pair_count <= 0;
            num_pairs <= 0;
            computation_complete <= 0;
        end else begin
            if (compute_start && enable) begin
                matrix_size <= num_simplices;
                reduction_state <= 4'b0001;
                current_col <= 0;
                pair_count <= 0;
                computation_complete <= 0;
                
                for (integer i = 0; i < num_simplices; i = i + 1) begin
                    for (integer j = 0; j < num_simplices; j = j + 1) begin
                        boundary_matrix[i][j] <= simplex_data[i] & simplex_data[j];
                    end
                    matrix_row[i] <= simplex_data[i];
                end
            end
            
            case (reduction_state)
                4'b0001: begin
                    if (current_col < matrix_size) begin
                        reduced_matrix[current_col] <= matrix_row[current_col];
                        reduction_state <= 4'b0010;
                    end else begin
                        reduction_state <= 4'b0101;
                    end
                end
                
                4'b0010: begin
                    current_row <= find_pivot(reduced_matrix[current_col], matrix_size);
                    if (find_pivot(reduced_matrix[current_col], matrix_size) < matrix_size) begin
                        reduction_state <= 4'b0011;
                    end else begin
                        current_col <= current_col + 1;
                        reduction_state <= 4'b0001;
                    end
                end
                
                4'b0011: begin
                    if (pivot_row[current_row] < MAX_SIMPLICES) begin
                        reduced_matrix[current_col] <= reduced_matrix[current_col] ^ reduced_matrix[pivot_row[current_row]];
                        reduction_state <= 4'b0010;
                    end else begin
                        pivot_row[current_row] <= current_col;
                        if (current_row != current_col) begin
                            persistence_pairs[pair_count][0] <= current_row;
                            persistence_pairs[pair_count][1] <= current_col;
                            pair_count <= pair_count + 1;
                        end
                        current_col <= current_col + 1;
                        reduction_state <= 4'b0001;
                    end
                end
                
                4'b0101: begin
                    num_pairs <= pair_count;
                    
                    for (integer dim = 0; dim <= max_dimension; dim = dim + 1) begin
                        betti_numbers[dim] <= 0;
                        for (integer i = 0; i < matrix_size; i = i + 1) begin
                            if (pivot_row[i] == MAX_SIMPLICES && (simplex_data[i] >> (dim * 4)) & 4'hF == dim) begin
                                betti_numbers[dim] <= betti_numbers[dim] + 1;
                            end
                        end
                        for (integer i = 0; i < pair_count; i = i + 1) begin
                            if ((simplex_data[persistence_pairs[i][0]] >> (dim * 4)) & 4'hF == dim) begin
                                betti_numbers[dim] <= betti_numbers[dim] - 1;
                            end
                        end
                    end
                    
                    computation_complete <= 1;
                    reduction_state <= 4'b0110;
                end
                
                4'b0110: begin
                    if (!compute_start) begin
                        reduction_state <= 4'b0000;
                    end
                end
            endcase
        end
    end

endmodule 