std_msgs/Header header
string[] dof_names

float64[] qpos
float64[] qvel
float64[] qacc

float64[] qfrc_applied
float64[] qfrc_actuator
float64[] qfrc_passive
float64[] qfrc_bias

float64[] qfrc_constraint
float64[] qfrc_constraint_equality
float64[] qfrc_constraint_friction
float64[] qfrc_constraint_limit
float64[] qfrc_constraint_contact
float64[] qfrc_constraint_other

uint32 velocity_pid_diagnostic_schema_version
string[] velocity_pid_joint_names
float64[] velocity_pid_target
float64[] velocity_pid_actual
float64[] velocity_pid_error
float64[] velocity_pid_p_term
float64[] velocity_pid_i_term
float64[] velocity_pid_candidate_i_term
float64[] velocity_pid_peak_abs_candidate_i_term
float64[] velocity_pid_d_term
float64[] velocity_pid_raw_output
float64[] velocity_pid_effort_clamped_output
bool[] velocity_pid_i_clamp_active
bool[] velocity_pid_effort_clamp_active
bool[] velocity_pid_i_clamp_ever
bool[] velocity_pid_effort_clamp_ever
uint64[] velocity_pid_control_update_count
float64[] velocity_pid_kp
float64[] velocity_pid_ki
float64[] velocity_pid_kd
float64[] velocity_pid_i_max
float64[] velocity_pid_i_min
bool[] velocity_pid_antiwindup
float64[] velocity_pid_effort_limit

int32 nefc
int32 ncon
