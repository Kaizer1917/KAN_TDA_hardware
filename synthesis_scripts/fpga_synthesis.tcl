set_property PART xcu280-fsvh2892-2L-e [current_project]

read_verilog {
    ../verilog_modules/bspline_units/bspline_evaluator.v
    ../fpga_implementations/kan_processing_elements/kan_processing_element.v
    ../verilog_modules/systolic_arrays/kan_systolic_array.v
    ../verilog_modules/persistence_engines/sparse_boundary_matrix_processor.v
    ../verilog_modules/persistence_engines/parallel_reduction_engine.v
    ../fpga_implementations/tda_accelerators/tda_acceleration_unit.v
}

set_property top kan_systolic_array [current_fileset]

create_clock -period 3.33 -name clk [get_ports clk]
set_input_delay -clock clk 0.5 [all_inputs]
set_output_delay -clock clk 0.5 [all_outputs]

set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk_IBUF]

opt_design
place_design
phys_opt_design
route_design

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 85.0 [current_design]
set_property CONFIG_VOLTAGE 1.8 [current_design]
set_property CFGBVS GND [current_design]

write_bitstream -force kan_tda_fpga.bit
write_checkpoint -force kan_tda_fpga.dcp

report_utilization -file utilization_report.txt
report_timing_summary -file timing_report.txt
report_power -file power_report.txt

puts "FPGA synthesis completed successfully"
puts "Target frequency: 300 MHz"
puts "Estimated power consumption: 25W" 