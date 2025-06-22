create_project kan_tda_fpga ./kan_tda_fpga -part xcu250-figd2104-2L-e -force

add_files -fileset sources_1 {
    ../fpga_implementations/kan_processing_elements/kan_processing_element.v
    ../fpga_implementations/tda_accelerators/tda_acceleration_unit.v
    ../verilog_modules/bspline_units/bspline_evaluator.v
    ../verilog_modules/persistence_engines/parallel_reduction_engine.v
}

set_property top tda_acceleration_unit [current_fileset]

create_clock -period 5.000 -name clk -waveform {0.000 2.500} [get_ports clk]

set_property PACKAGE_PIN AF14 [get_ports clk]
set_property IOSTANDARD LVDS [get_ports clk]

set_property PACKAGE_PIN AE13 [get_ports rst_n]
set_property IOSTANDARD LVCMOS18 [get_ports rst_n]

set_property PACKAGE_PIN AG14 [get_ports enable]
set_property IOSTANDARD LVCMOS18 [get_ports enable]

synth_design -top tda_acceleration_unit -part xcu250-figd2104-2L-e

opt_design
place_design
phys_opt_design
route_design

report_timing_summary -file timing_summary.rpt
report_utilization -file utilization.rpt
report_power -file power.rpt

write_checkpoint -force post_route.dcp
write_bitstream -force kan_tda_fpga.bit

close_project 