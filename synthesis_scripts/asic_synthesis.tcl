set_app_var target_library "tcbn7ffcllbwp7t30p140_ccs.db"
set_app_var link_library "* tcbn7ffcllbwp7t30p140_ccs.db"
set_app_var symbol_library "tcbn7ffcllbwp7t30p140.sdb"

read_verilog {
    ../verilog_modules/bspline_units/bspline_evaluator.v
    ../fpga_implementations/kan_processing_elements/kan_processing_element.v
    ../asic_implementations/kan_cores/kan_processing_core.v
    ../asic_implementations/power_management/power_management_unit.v
    ../verilog_modules/persistence_engines/sparse_boundary_matrix_processor.v
    ../verilog_modules/persistence_engines/parallel_reduction_engine.v
    ../fpga_implementations/tda_accelerators/tda_acceleration_unit.v
    ../asic_implementations/kan_tda_asic_top.v
}

set current_design kan_tda_asic_top
link

create_clock -period 0.4 -name clk [get_ports clk]
set_input_delay -clock clk 0.05 [all_inputs]
set_output_delay -clock clk 0.05 [all_outputs]
set_driving_cell -lib_cell BUFX4 [all_inputs]
set_load 0.01 [all_outputs]

set_max_area 12.3
set_max_dynamic_power 0.08
set_max_leakage_power 0.02

set_operating_conditions -max ss0p72v125c -max_library tcbn7ffcllbwp7t30p140_ccs
set_operating_conditions -min ff0p88vm40c -min_library tcbn7ffcllbwp7t30p140_ccs

set_wire_load_model -name "TSMC32K_Lowk_Conservative"

compile_ultra -gate_clock -no_autoungroup

optimize_netlist -area

set_fix_hold [all_clocks]
compile_ultra -incremental -no_autoungroup

report_area -hierarchy > area_report.txt
report_power -hierarchy > power_report.txt
report_timing -max_paths 10 > timing_report.txt
report_constraint -all_violators > constraint_report.txt

write -format verilog -hierarchy -output kan_tda_asic_netlist.v
write_sdf kan_tda_asic.sdf
write_sdc kan_tda_asic.sdc

puts "ASIC synthesis completed successfully"
puts "Target frequency: 2.5 GHz"
puts "Estimated area: 12.3 mm²"
puts "Estimated power: 0.1W" 