# https://github.com/osrf/lrauv/wiki/Message-definitions#lrauvstate
std_msgs/Header header
float32 prop_omega
float32 prop_thrust
float32 prop_torque
float32 rudder_angle
float32 elevator_angle
float32 mass_position
float32 buoyancy_position
float32 depth
float32 speed
float64 latitude_deg
float64 longitude_deg
float32 net_buoy
geometry_msgs/Vector3 pos
geometry_msgs/Vector3 pos_rph
geometry_msgs/Vector3 pos_dot
geometry_msgs/Vector3 rate_uvw
geometry_msgs/Vector3 rate_pqr
float32 north_current
float32 east_current
float32 temperature
float32 salinity
float32 density
float64 battery_voltage
float64 battery_current
float64 battery_charge
float64 battery_percentage