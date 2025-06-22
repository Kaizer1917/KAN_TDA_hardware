set design_name "kan_tda_asic_core"
set target_library "tcbn7ffcllbwp7t30p140_ccs.db"
set link_library "* $target_library"
set symbol_library "tcbn7ffcllbwp7t30p140.sdb"

set search_path ". /tools/synopsys/libraries/TSMC7nm"

set hdlin_translate_off_skip_text true
set hdlin_enable_presto_for_verilog true
set compile_seqmap_propagate_constants false

read_verilog [list \
    ../asic_designs/kan_tda_asic_core.v \
    ../verilog_modules/interconnect/axi4_interconnect.v \
    ../verilog_modules/kan_processing_units/kan_multi_core.v \
    ../verilog_modules/memory_controllers/cache_controller.v \
    ../verilog_modules/power_management/clock_gating_unit.v \
    ../verilog_modules/tda_accelerators/homology_engine.v \
    ../../asic_implementations/kan_cores/kan_processing_core.v \
    ../../asic_implementations/power_management/power_management_unit.v \
    ../../verilog_modules/bspline_units/bspline_evaluator.v \
    ../../verilog_modules/persistence_engines/parallel_reduction_engine.v \
    ../../verilog_modules/systolic_arrays/kan_systolic_array.v \
]

current_design $design_name

link

set_wire_load_model -name "tsmc7nm_wlm" -library $target_library

create_clock -name "sys_clk" -period 0.4 [get_ports sys_clk]
create_clock -name "jtag_clk" -period 10.0 [get_ports jtag_tck]

set_clock_groups -asynchronous -group {sys_clk} -group {jtag_clk}

set_input_delay -clock sys_clk -max 0.1 [all_inputs]
set_input_delay -clock sys_clk -min 0.05 [all_inputs]
set_output_delay -clock sys_clk -max 0.1 [all_outputs]
set_output_delay -clock sys_clk -min 0.05 [all_outputs]

set_driving_cell -library $target_library -lib_cell BUFFD2BWP7T30P140 [all_inputs]
set_load [load_of tcbn7ffcllbwp7t30p140/BUFFD2BWP7T30P140/A] [all_outputs]

set_max_transition 0.15 [current_design]
set_max_fanout 16 [current_design]
set_max_capacitance 0.5 [current_design]

set_power_prediction true
set_dynamic_optimization true

set_operating_conditions -min_library $target_library -min "BCCOM" \
                        -max_library $target_library -max "WCCOM"

set_dont_use [get_lib_cells */*X0P*]
set_dont_use [get_lib_cells */*X0P5*]

set_ideal_network [get_ports por_rst_n]
set_ideal_network [get_ports jtag_trst_n]

set_case_analysis 0 [get_ports test_mode]
set_case_analysis 0 [get_ports scan_enable]

set_multicycle_path -setup 2 -from [get_clocks sys_clk] -to [get_clocks sys_clk] -through [get_pins *cache*/*]
set_multicycle_path -hold 1 -from [get_clocks sys_clk] -to [get_clocks sys_clk] -through [get_pins *cache*/*]

set_false_path -from [get_ports por_rst_n]
set_false_path -from [get_ports jtag_trst_n]
set_false_path -from [get_ports config_data[*]] -to [get_clocks sys_clk]

compile_ultra -no_autoungroup -no_boundary_optimization

optimize_netlist -area

set_critical_range 0.2 [current_design]
group_path -name "INPUTS" -from [all_inputs]
group_path -name "OUTPUTS" -to [all_outputs]
group_path -name "COMBO" -from [all_inputs] -to [all_outputs]

report_timing -path_type full_clock_expanded -delay_type max -max_paths 10 -nworst 2 -format {hpin arc cell delay arrival required} > timing_max.rpt
report_timing -path_type full_clock_expanded -delay_type min -max_paths 10 -nworst 2 -format {hpin arc cell delay arrival required} > timing_min.rpt

report_area -hierarchy > area.rpt
report_power -analysis_effort medium > power.rpt
report_constraints -all_violators > constraints.rpt
report_design > design.rpt

check_design > check_design.rpt
check_timing > check_timing.rpt

write -hierarchy -format verilog -output ${design_name}_syn.v
write_sdf -version 3.0 ${design_name}.sdf
write_sdc ${design_name}.sdc

set_dp_smartgen_options -all_options
create_dp_smartgen_constraints -exclude_ip_domains

saif_map -start

report_congestion > congestion.rpt
report_threshold_voltage_group > tvg.rpt

extract_physical_constraints ${design_name}.def

puts "ASIC synthesis completed successfully"
puts "Synthesized netlist: ${design_name}_syn.v"
puts "SDF file: ${design_name}.sdf"
puts "SDC constraints: ${design_name}.sdc"

exit 