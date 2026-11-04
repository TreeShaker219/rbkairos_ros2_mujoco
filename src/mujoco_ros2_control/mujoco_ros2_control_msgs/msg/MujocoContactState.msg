std_msgs/Header header
int32 ncon

int32[] contact_id
int32[] geom1_id
int32[] geom2_id
string[] geom1_name
string[] geom2_name

int32[] dim
int32[] efc_address
float64[] dist

# Flattened arrays, one contact after another.
float64[] position_xyz
float64[] frame_3x3
float64[] friction
float64[] force_local_6d
