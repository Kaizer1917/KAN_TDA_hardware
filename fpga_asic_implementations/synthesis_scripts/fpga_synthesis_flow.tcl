set design_name "kan_tda_fpga_top"
set top_module "kan_tda_fpga_top"
set target_part "xcvu9p-flga2104-2-i"
set board_part "xilinx.com:vcu118:part0:2.4"

create_project $design_name . -part $target_part -force
set_property board_part $board_part [current_project]

add_files [glob *.v]
add_files ../verilog_modules/interconnect/axi4_interconnect.v
add_files ../verilog_modules/kan_processing_units/kan_multi_core.v
add_files ../verilog_modules/memory_controllers/cache_controller.v
add_files ../verilog_modules/power_management/clock_gating_unit.v
add_files ../verilog_modules/tda_accelerators/homology_engine.v
add_files ../../fpga_implementations/interconnect/noc_router.v
add_files ../../fpga_implementations/interconnect/crossbar_switch.v
add_files ../../fpga_implementations/memory_controllers/ddr4_controller.v
add_files ../../fpga_implementations/kan_processing_elements/kan_processing_element.v
add_files ../../fpga_implementations/tda_accelerators/tda_acceleration_unit.v
add_files ../../verilog_modules/bspline_units/bspline_evaluator.v
add_files ../../verilog_modules/persistence_engines/parallel_reduction_engine.v
add_files ../../verilog_modules/systolic_arrays/kan_systolic_array.v

set_property top $top_module [current_fileset]

create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name clk_wiz_0
set_property -dict [list CONFIG.PRIMITIVE {PLL} CONFIG.PRIM_IN_FREQ {100.000} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {200.000} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {400.000} CONFIG.USE_RESET {true} CONFIG.USE_LOCKED {true}] [get_ips clk_wiz_0]

create_ip -name ddr4 -vendor xilinx.com -library ip -version 2.2 -module_name ddr4_0
set_property -dict [list CONFIG.C0.DDR4_TimePeriod {833} CONFIG.C0.DDR4_InputClockPeriod {4000} CONFIG.C0.DDR4_MemoryPart {MT40A512M16HA-083E} CONFIG.C0.DDR4_DataWidth {64} CONFIG.C0.DDR4_DataMask {DM_NO_DBI} CONFIG.C0.DDR4_Ecc {false} CONFIG.C0.DDR4_AxiSelection {true} CONFIG.C0.DDR4_AxiDataWidth {256} CONFIG.C0.DDR4_AxiAddressWidth {32}] [get_ips ddr4_0]

generate_target all [get_ips]

create_clock -period 10.000 [get_ports sys_clk_p]
set_property IOSTANDARD DIFF_SSTL12 [get_ports sys_clk_p]
set_property IOSTANDARD DIFF_SSTL12 [get_ports sys_clk_n]

create_clock -period 5.000 [get_clocks clk_out1_clk_wiz_0]
create_clock -period 2.500 [get_clocks clk_out2_clk_wiz_0]

set_clock_groups -asynchronous -group [get_clocks sys_clk_p] -group [get_clocks clk_out1_clk_wiz_0]
set_clock_groups -asynchronous -group [get_clocks sys_clk_p] -group [get_clocks clk_out2_clk_wiz_0]

set_property IOSTANDARD LVCMOS18 [get_ports {status_leds[*]}]
set_property IOSTANDARD LVCMOS18 [get_ports {user_buttons[*]}]
set_property IOSTANDARD LVCMOS18 [get_ports sys_rst_n]
set_property IOSTANDARD LVCMOS18 [get_ports processing_complete]
set_property IOSTANDARD LVCMOS18 [get_ports error_flag]

set_property PACKAGE_PIN AY24 [get_ports {status_leds[0]}]
set_property PACKAGE_PIN AY25 [get_ports {status_leds[1]}]
set_property PACKAGE_PIN BA27 [get_ports {status_leds[2]}]
set_property PACKAGE_PIN BA28 [get_ports {status_leds[3]}]
set_property PACKAGE_PIN BB26 [get_ports {status_leds[4]}]
set_property PACKAGE_PIN BB27 [get_ports {status_leds[5]}]
set_property PACKAGE_PIN BA25 [get_ports {status_leds[6]}]
set_property PACKAGE_PIN BB25 [get_ports {status_leds[7]}]

set_property PACKAGE_PIN BD23 [get_ports {user_buttons[0]}]
set_property PACKAGE_PIN BE23 [get_ports {user_buttons[1]}]
set_property PACKAGE_PIN BE22 [get_ports {user_buttons[2]}]
set_property PACKAGE_PIN BC22 [get_ports {user_buttons[3]}]

set_property PACKAGE_PIN BD22 [get_ports sys_rst_n]
set_property PACKAGE_PIN BE24 [get_ports processing_complete]
set_property PACKAGE_PIN BF24 [get_ports error_flag]

set_max_delay -from [get_clocks clk_out1_clk_wiz_0] -to [get_clocks clk_out2_clk_wiz_0] 2.000
set_max_delay -from [get_clocks clk_out2_clk_wiz_0] -to [get_clocks clk_out1_clk_wiz_0] 2.000

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 1.8 [current_design]

synth_design -top $top_module -part $target_part
write_checkpoint -force post_synth.dcp

opt_design
place_design
phys_opt_design
route_design
phys_opt_design

write_checkpoint -force post_route.dcp

report_timing_summary -file timing_summary.rpt
report_utilization -file utilization.rpt
report_power -file power.rpt

write_bitstream -force $design_name.bit

puts "FPGA synthesis and implementation completed successfully"
puts "Bitstream generated: $design_name.bit"

close_project 