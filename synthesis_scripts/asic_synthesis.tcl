set search_path [list . $search_path]
set target_library "typical.db"
set link_library "* $target_library"

set design_name "tda_asic_top"
set clock_period 2.0
set clock_uncertainty 0.1
set input_delay 0.2
set output_delay 0.2

read_verilog -r {
    ../asic_implementations/kan_cores/kan_processing_core.v
    ../asic_implementations/power_management/power_management_unit.v
    ../asic_implementations/tda_units/tda_asic_top.v
    ../verilog_modules/bspline_units/bspline_evaluator.v
    ../verilog_modules/persistence_engines/parallel_reduction_engine.v
}

current_design $design_name

create_clock -name "clk" -period $clock_period [get_ports clk]
set_clock_uncertainty $clock_uncertainty [get_clocks clk]

set_input_delay $input_delay -clock clk [all_inputs]
set_output_delay $output_delay -clock clk [all_outputs]

set_driving_cell -lib_cell BUFX2_LVT [all_inputs]
set_load 0.1 [all_outputs]

set_max_area 0
set_max_dynamic_power 50
set_max_leakage_power 5

compile_ultra -gate_clock
optimize_netlist -area

ungroup -all -flatten
compile_ultra -incremental

check_design
report_qor
report_timing
report_area
report_power
report_constraint

write_sdf -version 2.1 ${design_name}.sdf
write -format verilog -hierarchy -output ${design_name}_syn.v
write_sdc ${design_name}.sdc

quit 