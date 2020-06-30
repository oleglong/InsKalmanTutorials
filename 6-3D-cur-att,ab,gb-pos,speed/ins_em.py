import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ins_sig_gen import generate_signals
from ins_ekf import ins_ext_kfilter
matplotlib.use('Agg')


# Config
imu_period = 0.005
imu_init_time = 2
accel_bias_std = 0.3
accel_bias0 = np.matrix([
	# X
	[ 0.03 ],
	# Y
	[ -0.07 ],
	# Z
	[ -0.04 ]
])
accel_w_std = 0.05
gyro_bias_std = np.deg2rad( 1.0 )
gyro_bias0 = np.matrix([
	# X
	[ np.deg2rad( 0.9 ) ],
	# Y
	[ np.deg2rad( 1.1 ) ],
	# Z
	[ np.deg2rad( -0.5 ) ]
])
gyro_w_std = np.deg2rad( 0.2 )
imu_attitude0_std = np.deg2rad( 1.0 )
imu_attitude0_err = np.matrix([
	# Psi
	[ np.deg2rad( -0.4 ) ],
	# Theta
	[ np.deg2rad( +0.2 ) ],
	# Gamma
	[ np.deg2rad( 0.3 ) ]
])
gnss_period = 0.25
gnss_speed_w_std = 0.2
gnss_dist_w_std = 0.5

gnss_off_time = 50

speed_changes = [
	[ +0,    imu_init_time ],
	
	[ +3,    3 ],	
	[ +0,    6 ],	
	[ +2,    5 ],
	[ +0,    4 ],	
	[ -3,    4 ],
	[ +0,    15 ],	
	[ +2,    4 ],	
	[ +0,    10 ],
	[ +0.5,  5 ],	
	[ +0,    5 ]
]
rot_changes_x = [
	[ np.deg2rad( +0 ),    imu_init_time ],
		
	# 1
	[ np.deg2rad( +0 ),    3 ],
	
	# 2
	[ np.deg2rad( +20 ),   4 ],
	# 3
	[ np.deg2rad( -20 ),   4 ],
	
	# 4
	[ np.deg2rad( +0 ),    8 ],
	
	# 5	
	[ np.deg2rad( +0 ),    3 ],
	# 6	
	[ np.deg2rad( +0 ),    1 ],
	# 7
	[ np.deg2rad( +0 ),    3 ],
	
	# 8
	[ np.deg2rad( +0 ),    9 ],
	
	# 9
	[ np.deg2rad( +0 ),    2 ],
	# 10
	[ np.deg2rad( +0 ),    2 ],
	
	# 11	
	[ np.deg2rad( +0 ),    6 ],
	# 12	
	[ np.deg2rad( +0 ),    7 ],
	
	# 13
	[ np.deg2rad( +0 ),    2 ],
	# 14
	[ np.deg2rad( -0 ),    2 ],
	
	# 15	
	[ np.deg2rad( +0 ),    5 ]
]
rot_changes_y = [
	[ np.deg2rad( +0 ),    imu_init_time ],
		
	# 1
	[ np.deg2rad( +0 ),    3 ],
	
	# 2
	[ np.deg2rad( +0 ),    4 ],
	# 3
	[ np.deg2rad( +0 ),    4 ],
	
	# 4
	[ np.deg2rad( +80 ),   8 ],
	
	# 5	
	[ np.deg2rad( +0 ),    3 ],
	# 6	
	[ np.deg2rad( +0 ),    1 ],
	# 7
	[ np.deg2rad( +0 ),    3 ],
	
	# 8	
	[ np.deg2rad( +70 ),   9 ],
	
	# 9
	[ np.deg2rad( +0 ),    2 ],
	# 10
	[ np.deg2rad( +0 ),    2 ],
	
	# 11	
	[ np.deg2rad( -60 ),   6 ],
	# 12
	[ np.deg2rad( -70 ),   7 ],
	
	# 13
	[ np.deg2rad( +0 ),    2 ],
	# 14
	[ np.deg2rad( -0 ),    2 ],
	
	# 15	
	[ np.deg2rad( +0 ),    5 ]
]
rot_changes_z = [
	[ np.deg2rad( +0 ),    imu_init_time ],
		
	# 1
	[ np.deg2rad( +0 ),    3 ],
	
	# 2
	[ np.deg2rad( +0 ),    4 ],
	# 3
	[ np.deg2rad( +0 ),    4 ],
	
	# 4
	[ np.deg2rad( +0 ),    8 ],
	
	# 5	
	[ np.deg2rad( +20 ),   3 ],
	# 6	
	[ np.deg2rad( +0 ),    1 ],
	# 7
	[ np.deg2rad( -20 ),   3 ],
	
	# 8	
	[ np.deg2rad( +0 ),    9 ],
	
	# 9
	[ np.deg2rad( -20 ),   2 ],
	# 10
	[ np.deg2rad( +20 ),   2 ],
	
	# 11	
	[ np.deg2rad( +0 ),    6 ],
	# 12	
	[ np.deg2rad( +0 ),    7 ],
	
	# 13
	[ np.deg2rad( +15 ),   2 ],
	# 14
	[ np.deg2rad( -15 ),   2 ],
	
	# 15	
	[ np.deg2rad( +0 ),    5 ]
]


real_attitude0 = np.matrix([
	# Psi
	[ 0 ],
	# Theta
	[ 0 ],
	# Gamma
	[ 0 ]
])

# Generate INS signals
[ imu_time, imu_accel, imu_gyro, 
  gnss_time, gnss_speed, gnss_dist,
  real_accel_bias, real_gyro_bias, 
  real_glob_attitude, real_glob_accel, real_glob_speed, real_glob_speed_norm, real_glob_dist 
] = generate_signals( 
	speed_changes, rot_changes_x, rot_changes_y, rot_changes_z, real_attitude0,
	imu_period, accel_bias0, accel_w_std, gyro_bias0, gyro_w_std,
	gnss_period, gnss_speed_w_std, gnss_dist_w_std
)

# Simulate GNSS off
off_cnt = int( gnss_off_time / gnss_period ) + 1
gnss_time = gnss_time[ : off_cnt ]
gnss_speed = gnss_speed[ : off_cnt ]
gnss_dist = gnss_dist[ : off_cnt ]

print(len(imu_accel))
# Initial params estimation
imu_attitude0 = real_attitude0 + np.matrix([
	# Psi error
	[ imu_attitude0_err.item( (0, 0) ) ],
	# Theta error
	[ imu_attitude0_err.item( (1, 0) ) ],
	# Gamma error
	[ imu_attitude0_err.item( (2, 0) ) ]
])
print( 'Real psi0: '          + str( np.rad2deg( real_attitude0.item( ( 0, 0 ) ) ) ) + 
	   ', estimated psi0: '   + str( np.rad2deg( imu_attitude0.item(  ( 0, 0 ) ) ) ) )
print( 'Real theta0: '        + str( np.rad2deg( real_attitude0.item( ( 1, 0 ) ) ) ) + 
	   ', estimated theta0: ' + str( np.rad2deg( imu_attitude0.item(  ( 1, 0 ) ) ) ) )
print( 'Real gamma0: '        + str( np.rad2deg( real_attitude0.item( ( 2, 0 ) ) ) ) + 
	   ', estimated gamma0: ' + str( np.rad2deg( imu_attitude0.item(  ( 2, 0 ) ) ) ) )
	   
imu_gyro_bias0 = np.matrix([ [0], [0], [0] ])
imu_init_cnt = int(imu_init_time/imu_period)
for gyro in imu_gyro[0 : imu_init_cnt]:
	imu_gyro_bias0 = imu_gyro_bias0 + gyro
imu_gyro_bias0 = imu_gyro_bias0 / imu_init_cnt

print( 'Real gyro bias x: '        + str( np.rad2deg( real_gyro_bias[ 0 ].item( ( 0, 0 ) ) ) ) + 
	   ', estimated gyro bias x: ' + str( np.rad2deg( imu_gyro_bias0.item( ( 0, 0 ) ) ) ) )
print( 'Real gyro bias y: '        + str( np.rad2deg( real_gyro_bias[ 0 ].item( ( 1, 0 ) ) ) ) + 
	   ', estimated gyro bias y: ' + str( np.rad2deg( imu_gyro_bias0.item( ( 1, 0 ) ) ) ) )
print( 'Real gyro bias z: '        + str( np.rad2deg( real_gyro_bias[ 0 ].item( ( 2, 0 ) ) ) ) + 
	   ', estimated gyro bias z: ' + str( np.rad2deg( imu_gyro_bias0.item( ( 2, 0 ) ) ) ) )

# Estimate body motion
[ ins_state, ins_var ] = ins_ext_kfilter( 
	imu_time, imu_accel, imu_gyro, accel_bias_std, accel_w_std, gyro_bias_std, gyro_w_std,
	imu_attitude0, imu_attitude0_std, imu_gyro_bias0, 
	gnss_time, gnss_speed, gnss_dist, gnss_speed_w_std, gnss_dist_w_std )

# IMU signals
imu_accel_x 			= [ v.item( (0, 0) ) for v in imu_accel ]
imu_accel_y 			= [ v.item( (1, 0) ) for v in imu_accel ]
imu_accel_z 			= [ v.item( (2, 0) ) for v in imu_accel ]
imu_rot_speed_x			= [ np.rad2deg( v.item( (0, 0) ) ) for v in imu_gyro ]
imu_rot_speed_y			= [ np.rad2deg( v.item( (1, 0) ) ) for v in imu_gyro ]
imu_rot_speed_z			= [ np.rad2deg( v.item( (2, 0) ) ) for v in imu_gyro ]
gnss_dist_x 			= [ v.item( (0, 0) ) for v in gnss_dist ]
gnss_dist_y 			= [ v.item( (1, 0) ) for v in gnss_dist ]
gnss_dist_z 			= [ v.item( (2, 0) ) for v in gnss_dist ]
gnss_speed_norm			= [ v.item( (0, 0) ) for v in gnss_speed ]

# Real signals
real_accel_bias_x 		= [ v.item( (0, 0) ) for v in real_accel_bias ]
real_accel_bias_y 		= [ v.item( (1, 0) ) for v in real_accel_bias ]
real_accel_bias_z 		= [ v.item( (2, 0) ) for v in real_accel_bias ]
real_gyro_bias_x 		= [ np.rad2deg( v.item( (0, 0) ) ) for v in real_gyro_bias ]
real_gyro_bias_y 		= [ np.rad2deg( v.item( (1, 0) ) ) for v in real_gyro_bias ]
real_gyro_bias_z 		= [ np.rad2deg( v.item( (2, 0) ) ) for v in real_gyro_bias ]
real_glob_accel_x 		= [ v.item( (0, 0) ) for v in real_glob_accel ]
real_glob_accel_y 		= [ v.item( (1, 0) ) for v in real_glob_accel ]
real_glob_accel_z 		= [ v.item( (2, 0) ) for v in real_glob_accel ]
real_glob_speed_x 		= [ v.item( (0, 0) ) for v in real_glob_speed ]
real_glob_speed_y 		= [ v.item( (1, 0) ) for v in real_glob_speed ]
real_glob_speed_z 		= [ v.item( (2, 0) ) for v in real_glob_speed ]
real_glob_speed_norm	= [ v.item( (0, 0) ) for v in real_glob_speed_norm ]
real_glob_dist_x 		= [ v.item( (0, 0) ) for v in real_glob_dist ]
real_glob_dist_y 		= [ v.item( (1, 0) ) for v in real_glob_dist ]
real_glob_dist_z 		= [ v.item( (2, 0) ) for v in real_glob_dist ]
real_psi				= [ np.rad2deg( v.item( (0, 0) ) ) for v in real_glob_attitude ]
real_theta				= [ np.rad2deg( v.item( (1, 0) ) ) for v in real_glob_attitude ]
real_gamma				= [ np.rad2deg( v.item( (2, 0) ) ) for v in real_glob_attitude ]

# Estimated signals
ekf_glob_dist_x			= [ v.item( ( 0, 0 ) ) 	for v in ins_state ]
ekf_glob_dist_x_var		= [ np.sqrt( v.item( ( 0, 0 ) ) ) for v in ins_var ]
ekf_glob_dist_y			= [ v.item( ( 1, 0 ) ) 	for v in ins_state ]
ekf_glob_dist_y_var		= [ np.sqrt( v.item( ( 1, 1 ) ) ) for v in ins_var ]
ekf_glob_dist_z			= [ v.item( ( 2, 0 ) ) 	for v in ins_state ]
ekf_glob_dist_z_var		= [ np.sqrt( v.item( ( 2, 2 ) ) ) for v in ins_var ]
ekf_glob_speed_x		= [ v.item( ( 3, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_x_var	= [ np.sqrt( v.item( ( 3, 3 ) ) ) for v in ins_var ]
ekf_glob_speed_y		= [ v.item( ( 4, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_y_var	= [ np.sqrt( v.item( ( 4, 4 ) ) ) for v in ins_var ]
ekf_glob_speed_z		= [ v.item( ( 5, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_z_var	= [ np.sqrt( v.item( ( 5, 5 ) ) ) for v in ins_var ]
ekf_glob_speed_norm		= [ np.sqrt( x**2 + y**2 + z**2 ) for x, y, z in zip( ekf_glob_speed_x, ekf_glob_speed_y, ekf_glob_speed_z ) ]
ekf_accel_bias_x		= [ v.item( ( 6, 0 ) ) 	for v in ins_state ]
ekf_accel_bias_x_var	= [ np.sqrt( v.item( ( 6, 6 ) ) ) for v in ins_var ]
ekf_accel_bias_y		= [ v.item( ( 7, 0 ) ) 	for v in ins_state ]
ekf_accel_bias_y_var	= [ np.sqrt( v.item( ( 7, 7 ) ) ) for v in ins_var ]
ekf_accel_bias_z		= [ v.item( ( 8, 0 ) ) 	for v in ins_state ]
ekf_accel_bias_z_var	= [ np.sqrt( v.item( ( 8, 8 ) ) ) for v in ins_var ]
ekf_gyro_bias_x			= [ np.rad2deg( v.item( ( 9, 0 ) ) ) 	for v in ins_state ]
ekf_gyro_bias_x_var		= [ np.rad2deg( np.sqrt( v.item( ( 9, 9 ) ) ) ) for v in ins_var ]
ekf_gyro_bias_y			= [ np.rad2deg( v.item( ( 10, 0 ) ) ) for v in ins_state ]
ekf_gyro_bias_y_var		= [ np.rad2deg( np.sqrt( v.item( ( 10, 10 ) ) ) ) for v in ins_var ]
ekf_gyro_bias_z			= [ np.rad2deg( v.item( ( 11, 0 ) ) ) for v in ins_state ]
ekf_gyro_bias_z_var		= [ np.rad2deg( np.sqrt( v.item( ( 11, 11 ) ) ) ) for v in ins_var ]
ekf_psi					= [ np.rad2deg( v.item( ( 12, 0 ) ) ) for v in ins_state ]
ekf_psi_var				= [ np.rad2deg( np.sqrt( v.item( ( 12, 12 ) ) ) ) for v in ins_var ]
ekf_theta				= [ np.rad2deg( v.item( ( 13, 0 ) ) ) for v in ins_state ]
ekf_theta_var			= [ np.rad2deg( np.sqrt( v.item( ( 13, 13 ) ) ) ) for v in ins_var ]
ekf_gamma				= [ np.rad2deg( v.item( ( 14, 0 ) ) ) for v in ins_state ]
ekf_gamma_var			= [ np.rad2deg( np.sqrt( v.item( ( 14, 14 ) ) ) ) for v in ins_var ]


def plot_sectors(y):
	plt.text(gnss_off_time - 10, y, 'A', fontsize=12)
	plt.text(gnss_off_time + 10, y, 'B', fontsize=12)
	plt.axvline(x = gnss_off_time, color='k', linestyle=':')
	
def plot_save(fname):
	plt.savefig(fname, dpi=350)
	

	
# Errors and Variances
dist_x_err_a = real_glob_dist_x[ gnss_off_time ] - ekf_glob_dist_x[ gnss_off_time ]
dist_y_err_a = real_glob_dist_y[ gnss_off_time ] - ekf_glob_dist_y[ gnss_off_time ]
dist_z_err_a = real_glob_dist_z[ gnss_off_time ] - ekf_glob_dist_z[ gnss_off_time ]
spd_x_err_a = real_glob_speed_x[ gnss_off_time ] - ekf_glob_speed_x[ gnss_off_time ]
spd_y_err_a = real_glob_speed_y[ gnss_off_time ] - ekf_glob_speed_y[ gnss_off_time ]
spd_z_err_a = real_glob_speed_z[ gnss_off_time ] - ekf_glob_speed_z[ gnss_off_time ]
accel_bias_x_err_a = real_accel_bias_x[ gnss_off_time ] - ekf_accel_bias_x[ gnss_off_time ]
accel_bias_y_err_a = real_accel_bias_y[ gnss_off_time ] - ekf_accel_bias_y[ gnss_off_time ]
accel_bias_z_err_a = real_accel_bias_z[ gnss_off_time ] - ekf_accel_bias_z[ gnss_off_time ]
gyro_bias_x_err_a = real_gyro_bias_x[ gnss_off_time ] - ekf_gyro_bias_x[ gnss_off_time ]
gyro_bias_y_err_a = real_gyro_bias_y[ gnss_off_time ] - ekf_gyro_bias_y[ gnss_off_time ]
gyro_bias_z_err_a = real_gyro_bias_z[ gnss_off_time ] - ekf_gyro_bias_z[ gnss_off_time ]
psi_err_a = real_psi[ gnss_off_time ] - ekf_psi[ gnss_off_time ]
theta_err_a = real_theta[ gnss_off_time ] - ekf_theta[ gnss_off_time ]
gamma_err_a = real_gamma[ gnss_off_time ] - ekf_gamma[ gnss_off_time ]

dist_x_var_a = ekf_glob_dist_x_var[ gnss_off_time ]
dist_y_var_a = ekf_glob_dist_y_var[ gnss_off_time ]
dist_z_var_a = ekf_glob_dist_z_var[ gnss_off_time ]
spd_x_var_a = ekf_glob_speed_x_var[ gnss_off_time ]
spd_y_var_a = ekf_glob_speed_y_var[ gnss_off_time ]
spd_z_var_a = ekf_glob_speed_z_var[ gnss_off_time ]
accel_bias_x_var_a = ekf_accel_bias_x_var[ gnss_off_time ]
accel_bias_y_var_a = ekf_accel_bias_y_var[ gnss_off_time ]
accel_bias_z_var_a = ekf_accel_bias_z_var[ gnss_off_time ]
gyro_bias_x_var_a = np.rad2deg( ekf_gyro_bias_x_var[ gnss_off_time ] )
gyro_bias_y_var_a = np.rad2deg( ekf_gyro_bias_y_var[ gnss_off_time ] )
gyro_bias_z_var_a = np.rad2deg( ekf_gyro_bias_z_var[ gnss_off_time ] )
psi_var_a = ekf_psi_var[ gnss_off_time ]
theta_var_a = ekf_theta_var[ gnss_off_time ]
gamma_var_a = ekf_gamma_var[ gnss_off_time ]

dist_x_err_b = real_glob_dist_x[ -1 ] - ekf_glob_dist_x[ -1 ]
dist_y_err_b = real_glob_dist_y[ -1 ] - ekf_glob_dist_y[ -1 ]
dist_z_err_b = real_glob_dist_z[ -1 ] - ekf_glob_dist_z[ -1 ]
spd_x_err_b = real_glob_speed_x[ -1 ] - ekf_glob_speed_x[ -1 ]
spd_y_err_b = real_glob_speed_y[ -1 ] - ekf_glob_speed_y[ -1 ]
spd_z_err_b = real_glob_speed_z[ -1 ] - ekf_glob_speed_z[ -1 ]
accel_bias_x_err_b = real_accel_bias_x[ -1 ] - ekf_accel_bias_x[ -1 ]
accel_bias_y_err_b = real_accel_bias_y[ -1 ] - ekf_accel_bias_y[ -1 ]
accel_bias_z_err_b = real_accel_bias_z[ -1 ] - ekf_accel_bias_z[ -1 ]
gyro_bias_x_err_b = real_gyro_bias_x[ -1 ] - ekf_gyro_bias_x[ -1 ]
gyro_bias_y_err_b = real_gyro_bias_y[ -1 ] - ekf_gyro_bias_y[ -1 ]
gyro_bias_z_err_b = real_gyro_bias_z[ -1 ] - ekf_gyro_bias_z[ -1 ]
psi_err_b = real_psi[ -1 ] - ekf_psi[ -1 ]
theta_err_b = real_theta[ -1 ] - ekf_theta[ -1 ]
gamma_err_b = real_gamma[ -1 ] - ekf_gamma[ -1 ]

dist_x_var_b = ekf_glob_dist_x_var[ -1 ]
dist_y_var_b = ekf_glob_dist_y_var[ -1 ]
dist_z_var_b = ekf_glob_dist_z_var[ -1 ]
spd_x_var_b = ekf_glob_speed_x_var[ -1 ]
spd_y_var_b = ekf_glob_speed_y_var[ -1 ]
spd_z_var_b = ekf_glob_speed_z_var[ -1 ]
accel_bias_x_var_b = ekf_accel_bias_x_var[ -1 ]
accel_bias_y_var_b = ekf_accel_bias_y_var[ -1 ]
accel_bias_z_var_b = ekf_accel_bias_z_var[ -1 ]
gyro_bias_x_var_b = ekf_gyro_bias_x_var[ -1 ]
gyro_bias_y_var_b = ekf_gyro_bias_y_var[ -1 ]
gyro_bias_z_var_b = ekf_gyro_bias_z_var[ -1 ]
psi_var_b = ekf_psi_var[ -1 ]
theta_var_b = ekf_theta_var[ -1 ]
gamma_var_b = ekf_gamma_var[ -1 ]

print('\tName\tVal A\tDev A\tVal B\tDev B' )
print( '\tDst X\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( dist_x_err_a, dist_x_var_a, dist_x_err_b, dist_x_var_b ) )
print( '\tDst Y\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( dist_y_err_a, dist_y_var_a, dist_y_err_b, dist_y_var_b ) )
print( '\tDst Z\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( dist_z_err_a, dist_z_var_a, dist_z_err_b, dist_z_var_b ) )
print( '\tSpd X\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( spd_x_err_a, spd_x_var_a, spd_x_err_b, spd_x_var_b ) )
print( '\tSpd Y\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( spd_y_err_a, spd_y_var_a, spd_y_err_b, spd_y_var_b ) )
print( '\tSpd Z\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( spd_z_err_a, spd_z_var_a, spd_z_err_b, spd_z_var_b ) )
print( '\tBsA X\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( accel_bias_x_err_a, accel_bias_x_var_a, accel_bias_x_err_b, accel_bias_x_var_b ) )
print( '\tBsA Y\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( accel_bias_y_err_a, accel_bias_y_var_a, accel_bias_y_err_b, accel_bias_y_var_b ) )
print( '\tBsA Z\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( accel_bias_z_err_a, accel_bias_z_var_a, accel_bias_z_err_b, accel_bias_z_var_b ) )
print( '\tBsG X\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( gyro_bias_x_err_a, gyro_bias_x_var_a, gyro_bias_x_err_b, gyro_bias_x_var_b ) )
print( '\tBsG Y\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( gyro_bias_y_err_a, gyro_bias_y_var_a, gyro_bias_y_err_b, gyro_bias_y_var_b ) )
print( '\tBsG Z\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( gyro_bias_z_err_a, gyro_bias_z_var_a, gyro_bias_z_err_b, gyro_bias_z_var_b ) )
print( '\tPsi  \t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( psi_err_a, psi_var_a, psi_err_b, psi_var_b ) )
print( '\tTheta\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( theta_err_a, theta_var_a, theta_err_b, theta_var_b ) )
print( '\tGamma\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( gamma_err_a, gamma_var_a, gamma_err_b, gamma_var_b ) )

plt.figure()
#plt.suptitle('Angular speed (IMU)')
plt.plot(imu_time, imu_rot_speed_x, 'r')

plt.figure()
plt.plot(imu_time, imu_rot_speed_y, 'r')

plt.figure()
plt.plot(imu_time, imu_rot_speed_z, 'r')

plt.figure()
#plt.suptitle('Accel (IMU)')
plt.plot(imu_time, imu_accel_x, 'r')

plt.figure()
plt.plot(imu_time, imu_accel_y, 'r')

plt.figure()
plt.plot(imu_time, imu_accel_z, 'r')


plt.figure()
#plt.suptitle('Accel bias')
line1, = plt.plot( imu_time, real_accel_bias_x, 'b', linestyle=':', linewidth=3 )
line3, = plt.plot( imu_time, ekf_accel_bias_x,  'c', linestyle='-', linewidth=2 )
plot_sectors( 0.04 )
plt.legend( ( line1, line3 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/abias_x.png')

plt.figure()
line1, = plt.plot(imu_time, real_accel_bias_y, 'b', linestyle=':', linewidth=3 )
line3, = plt.plot(imu_time, ekf_accel_bias_y,  'c', linestyle='-', linewidth=2 )
plot_sectors( 0.00 )
plt.legend( ( line1, line3 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/abias_y.png')

plt.figure()
line1, = plt.plot(imu_time, real_accel_bias_z, 'b', linestyle=':', linewidth=3 )
line3, = plt.plot(imu_time, ekf_accel_bias_z,  'c', linestyle='-', linewidth=2 )
plot_sectors( 0.05 )
plt.legend( ( line1, line3 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/abias_z.png')

plt.figure()
plt.plot(imu_time, ekf_accel_bias_x_var, 'c')
plot_sectors( 0.1 )
plot_save('plots/abias_var_x.png')

plt.figure()
plt.plot(imu_time, ekf_accel_bias_y_var, 'c')
plot_sectors( 0.1 )
plot_save('plots/abias_var_y.png')

plt.figure()
plt.plot(imu_time, ekf_accel_bias_z_var, 'c')
plot_sectors( 0.1 )
plot_save('plots/abias_var_z.png')


plt.figure()
#plt.suptitle('Gyro bias')
line1, = plt.plot(imu_time, real_gyro_bias_x, 'b', linestyle=':', linewidth=3 )
line3, = plt.plot(imu_time, ekf_gyro_bias_x,  'c', linestyle='-', linewidth=2 )
plot_sectors( 0.92 )
plt.legend( ( line1, line3 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/gbias_x.png')

plt.figure()
line1, = plt.plot(imu_time, real_gyro_bias_y, 'b', linestyle=':', linewidth=3 )
line3, = plt.plot(imu_time, ekf_gyro_bias_y,  'c', linestyle='-', linewidth=2 )
plot_sectors( 1.06 )
plt.legend( ( line1, line3 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/gbias_y.png')

plt.figure()
line1, = plt.plot(imu_time, real_gyro_bias_z, 'b', linestyle=':', linewidth=3 )
line3, = plt.plot(imu_time, ekf_gyro_bias_z,  'c', linestyle='-', linewidth=2 )
plot_sectors( -0.45 )
plt.legend( ( line1, line3 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/gbias_z.png')

plt.figure()
plt.plot(imu_time, ekf_gyro_bias_x_var, 'c')
plot_sectors( 0.1 )
plot_save('plots/gbias_var_x.png')

plt.figure()
plt.plot(imu_time, ekf_gyro_bias_y_var, 'c')
plot_sectors( 0.1 )
plot_save('plots/gbias_var_y.png')

plt.figure()
plt.plot(imu_time, ekf_gyro_bias_z_var, 'c')
plot_sectors( 0.1 )
plot_save('plots/gbias_var_z.png')


plt.figure()
#plt.suptitle('Speed norm')
line1, = plt.plot(imu_time,  real_glob_speed_norm, 'b', linestyle=':', linewidth=3)
line2, = plt.plot(gnss_time, gnss_speed_norm,      'g', linestyle='--', linewidth=2)
line3, = plt.plot(imu_time,  ekf_glob_speed_norm,  'c', linestyle='-', linewidth=2)
plot_sectors( 0.1 )
plt.legend( ( line1, line2, line3 ), ( 'Real value', 'GNSS', 'Estimation' ) )
plot_save('plots/spd_mod.png')


plt.figure()
#plt.suptitle('Speed')
line1, = plt.plot(imu_time,  real_glob_speed_x, 'b', linestyle=':', linewidth=3)
line3, = plt.plot(imu_time,  ekf_glob_speed_x,  'c', linestyle='-', linewidth=2)
plot_sectors( 1.2 )
plt.legend( ( line1, line3 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/spd_x.png')

plt.figure()
line1, = plt.plot(imu_time,  real_glob_speed_y, 'b', linestyle=':', linewidth=3)
line3, = plt.plot(imu_time,  ekf_glob_speed_y,  'c', linestyle='-', linewidth=2)
plot_sectors( 0.5 )
plt.legend( ( line1, line3 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/spd_y.png')

plt.figure()
line1, = plt.plot(imu_time,  real_glob_speed_z, 'b', linestyle=':', linewidth=3)
line3, = plt.plot(imu_time,  ekf_glob_speed_z,  'c', linestyle='-', linewidth=2)
plot_sectors( -0.5 )
plt.legend( ( line1, line3 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/spd_z.png')


plt.figure()
#plt.suptitle('Position')
line1, = plt.plot(imu_time,  real_glob_dist_x, 'b', linestyle=':', linewidth=3)
line2, = plt.plot(gnss_time, gnss_dist_x,      'g', linestyle='--', linewidth=2)
line3, = plt.plot(imu_time,  ekf_glob_dist_x,  'c', linestyle='-', linewidth=2)
plot_sectors( 10 )
plt.legend( ( line1, line2, line3 ), ( 'Real value', 'GNSS', 'Estimation' ) )
plot_save('plots/pos_x.png')

plt.figure()
line1, = plt.plot(imu_time,  real_glob_dist_y, 'b', linestyle=':', linewidth=3)
line2, = plt.plot(gnss_time, gnss_dist_y,      'g', linestyle='--', linewidth=2)
line3, = plt.plot(imu_time,  ekf_glob_dist_y,  'c', linestyle='-', linewidth=2)
plot_sectors( 0 )
plt.legend( ( line1, line2, line3 ), ( 'Real value', 'GNSS', 'Estimation' ) )
plot_save('plots/pos_y.png')

plt.figure()
line1, = plt.plot(imu_time,  real_glob_dist_z, 'b', linestyle=':', linewidth=3)
line2, = plt.plot(gnss_time, gnss_dist_z,      'g', linestyle='--', linewidth=2)
line3, = plt.plot(imu_time,  ekf_glob_dist_z,  'c', linestyle='-', linewidth=2)
plot_sectors( -20 )
plt.legend( ( line1, line2, line3 ), ( 'Real value', 'GNSS', 'Estimation' ) )
plot_save('plots/pos_z.png')

fig = plt.figure()
plt.suptitle('3D map')
ax = fig.add_subplot(111, projection='3d')
# Y is vertical for IMU
#plt.axis('equal')
line1, = plt.plot(real_glob_dist_x, real_glob_dist_z, real_glob_dist_y, 'b', linestyle=':', linewidth=3)
#plt.axis('equal')
line2, = plt.plot(gnss_dist_x, gnss_dist_z, gnss_dist_y, 'g', linestyle='--', linewidth=2)
#plt.axis('equal')
line3, = plt.plot(ekf_glob_dist_x, ekf_glob_dist_z, ekf_glob_dist_y, 'c', linestyle='-', linewidth=2)
plt.legend( ( line1, line2, line3 ), ( 'Real value', 'GNSS', 'Estimation' ) )
plot_save('plots/3d.png')

#plt.suptitle('Attitude')
plt.figure()
line1, = plt.plot(imu_time, real_psi, 'b', linestyle=':')
line2, = plt.plot(imu_time, ekf_psi,  'c', linestyle='-' )
plt.legend( ( line1, line2 ), ( 'Real value', 'Estimation' ) )
plot_sectors( 5 )
plot_save('plots/psi.png')

plt.figure()
line1, = plt.plot(imu_time, real_theta, 'b', linestyle=':')
line2, = plt.plot(imu_time, ekf_theta,  'c', linestyle='-' )
plt.legend( ( line1, line2 ), ( 'Real value', 'Estimation' ) )
plot_sectors( 5 )
plot_save('plots/theta.png')

plt.figure()
line1, = plt.plot(imu_time, real_gamma, 'b', linestyle=':', linewidth=3)
line2, = plt.plot(imu_time, ekf_gamma,  'c', linestyle='-', linewidth=2)
plot_sectors( 5 )
plt.legend( ( line1, line2 ), ( 'Real value', 'Estimation' ) )
plot_save('plots/gamma.png')

plt.figure()
plt.plot(imu_time, ekf_psi_var, 'c' )
plot_sectors( 1.4 )
plot_save('plots/psi_var.png')

plt.figure()
plt.plot(imu_time, ekf_theta_var, 'c' )
plot_sectors( 0.3 )
plot_save('plots/theta_var.png')

plt.figure()
plt.plot(imu_time, ekf_gamma_var, 'c' )
plot_sectors( 0.3 )
plot_save('plots/gamma_var.png')

plt.show()