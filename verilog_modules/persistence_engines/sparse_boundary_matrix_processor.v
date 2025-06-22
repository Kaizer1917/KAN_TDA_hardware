module sparse_boundary_matrix_processor #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 12,
    parameter MAX_ENTRIES = 4096,
    parameter CSC_PTR_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] values_in,
    input wire [ADDR_WIDTH-1:0] row_indices_in,
    input wire [CSC_PTR_WIDTH-1:0] col_ptr_in,
    input wire [ADDR_WIDTH-1:0] read_addr,
    input wire write_enable,
    output reg [DATA_WIDTH-1:0] values_out,
    output reg [ADDR_WIDTH-1:0] row_indices_out,
    output reg data_valid
);

    reg [DATA_WIDTH-1:0] values_memory [0:MAX_ENTRIES-1];
    reg [ADDR_WIDTH-1:0] row_indices_memory [0:MAX_ENTRIES-1];
    reg [CSC_PTR_WIDTH-1:0] col_pointers [0:1023];
    reg [ADDR_WIDTH-1:0] current_addr;
    reg [2:0] access_state;
    reg [CSC_PTR_WIDTH-1:0] col_start, col_end;
    reg [ADDR_WIDTH-1:0] entry_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            values_out <= 0;
            row_indices_out <= 0;
            data_valid <= 0;
            current_addr <= 0;
            access_state <= 0;
            entry_count <= 0;
        end else begin
            if (write_enable) begin
                values_memory[current_addr] <= values_in;
                row_indices_memory[current_addr] <= row_indices_in;
                col_pointers[col_ptr_in] <= current_addr;
                current_addr <= current_addr + 1;
                entry_count <= entry_count + 1;
            end
            
            if (enable) begin
                case (access_state)
                    3'b000: begin
                        col_start <= col_pointers[read_addr];
                        col_end <= col_pointers[read_addr + 1];
                        current_addr <= col_pointers[read_addr];
                        access_state <= 3'b001;
                        data_valid <= 0;
                    end
                    3'b001: begin
                        if (current_addr < col_end) begin
                            values_out <= values_memory[current_addr];
                            row_indices_out <= row_indices_memory[current_addr];
                            data_valid <= 1;
                            current_addr <= current_addr + 1;
                        end else begin
                            access_state <= 3'b010;
                            data_valid <= 0;
                        end
                    end
                    3'b010: begin
                        access_state <= 3'b000;
                    end
                endcase
            end
        end
    end

endmodule 