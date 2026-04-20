# from rattlesnake.environment.modal_environment import (
#     ModalMetadata,
#     ModalInstructions,
#     ModalCommands,
# )
# from rattlesnake.environment.sine_environment import SineMetadata
# from rattlesnake.environment.sine_utilities import SineSpecification
import numpy as np


# region Modal
# def make_modal_environment_metadata(
#     hardware_metadata, environment_name=MODAL_ENVIRONMENT_NAME
# ):
#     channel_list_bools = [True, True, True, True, True, True]
#     sample_rate = hardware_metadata.sample_rate
#     samples_per_frame = 1000
#     averaging_type = "Linear"
#     num_averages = 30
#     averaging_coefficient = 0.1
#     frf_technique = "H1"
#     frf_window = "rectangle"
#     overlap_percent = 0
#     trigger_type = "Free Run"
#     accept_type = "Accept All"
#     wait_for_steady_state = 0
#     trigger_channel = 0
#     pretrigger_percent = 0
#     trigger_slope_positive = True
#     trigger_level_percent = 0
#     hysteresis_level_percent = 0
#     hysteresis_frame_percent = 0
#     signal_generator_type = "random"
#     signal_generator_level = 0.01
#     signal_generator_min_frequency = 0
#     signal_generator_max_frequency = 500
#     signal_generator_on_percent = 0
#     acceptance_function = None
#     reference_channel_indices = [3, 4]
#     response_channel_indices = [0, 1, 2, 5]
#     output_channel_indices = [3, 4, 5]
#     output_oversample = hardware_metadata.output_oversample
#     exponential_window_value_at_frame_end = 0.25

#     return ModalMetadata(
#         environment_name,
#         channel_list_bools,
#         sample_rate,
#         samples_per_frame,
#         averaging_type,
#         num_averages,
#         averaging_coefficient,
#         frf_technique,
#         frf_window,
#         overlap_percent,
#         trigger_type,
#         accept_type,
#         wait_for_steady_state,
#         trigger_channel,
#         pretrigger_percent,
#         trigger_slope_positive,
#         trigger_level_percent,
#         hysteresis_level_percent,
#         hysteresis_frame_percent,
#         signal_generator_type,
#         signal_generator_level,
#         signal_generator_min_frequency,
#         signal_generator_max_frequency,
#         signal_generator_on_percent,
#         acceptance_function,
#         reference_channel_indices,
#         response_channel_indices,
#         output_channel_indices,
#         output_oversample,
#         exponential_window_value_at_frame_end,
#     )


# endregion


# region Sine
# def make_sine_environment_metadata(
#     hardware_metadata, environment_name=SINE_ENVIRONMENT_NAME
# ):
#     channel_list_bools = [True, True, True, True, True, True]
#     sample_rate = hardware_metadata.sample_rate
#     samples_per_frame = 50
#     number_of_channels = 6
#     specification = SineSpecification(
#         name="Sine Tone 1",
#         start_time=0,
#         num_control=1,
#         num_breakpoints=2,
#     )

#     table = specification.breakpoint_table

#     # --- Breakpoint 0 ---
#     table[0]["frequency"] = 1
#     table[0]["sweep_type"] = 0  # 0 = linear
#     table[0]["sweep_rate"] = 1
#     table[0]["amplitude"][0] = 1
#     table[0]["phase"][0] = 0  # radians

#     # --- Breakpoint 1 ---
#     table[1]["frequency"] = 10  # you must set frequency
#     table[1]["amplitude"][0] = 1
#     table[1]["phase"][0] = 0  # radians

#     # Last breakpoint should not have sweep info (UI enforces this)
#     table[1]["sweep_type"] = 0
#     table[1]["sweep_rate"] = 1

#     # --- Disable warnings / aborts ---
#     table["warning"][:] = np.nan
#     table["abort"][:] = np.nan

#     specifications = [specification]
#     ramp_time = 0.5
#     buffer_blocks = 2
#     control_convergence = 0.15
#     update_drives_after_environment = False
#     phase_fit = False
#     allow_automatic_aborts = False
#     tracking_filter_type = 0
#     tracking_filter_cutoff = 0.15
#     tracking_filter_order = 2
#     vk_filter_order = 2
#     vk_filter_bandwidth = 2
#     vk_filter_blocksize = 1000
#     vk_filter_overlap = 0.15
#     control_python_script = None
#     control_python_class = None
#     control_python_parameters = ""
#     control_channel_indices = [1]
#     output_channel_indices = [3, 4, 5]
#     response_transformation_matrix = None
#     output_transformation_matrix = None

#     return SineMetadata(
#         environment_name=environment_name,
#         channel_list_bools=channel_list_bools,
#         sample_rate=sample_rate,
#         samples_per_frame=samples_per_frame,
#         number_of_channels=number_of_channels,
#         specifications=specifications,
#         ramp_time=ramp_time,
#         buffer_blocks=buffer_blocks,
#         control_convergence=control_convergence,
#         update_drives_after_environment=update_drives_after_environment,
#         phase_fit=phase_fit,
#         allow_automatic_aborts=allow_automatic_aborts,
#         tracking_filter_type=tracking_filter_type,
#         tracking_filter_cutoff=tracking_filter_cutoff,
#         tracking_filter_order=tracking_filter_order,
#         vk_filter_order=vk_filter_order,
#         vk_filter_bandwidth=vk_filter_bandwidth,
#         vk_filter_blocksize=vk_filter_blocksize,
#         vk_filter_overlap=vk_filter_overlap,
#         control_python_script=control_python_script,
#         control_python_class=control_python_class,
#         control_python_parameters=control_python_parameters,
#         control_channel_indices=control_channel_indices,
#         output_channel_indices=output_channel_indices,
#         response_transformation_matrix=response_transformation_matrix,
#         output_transformation_matrix=output_transformation_matrix,
#     )


# endregion
