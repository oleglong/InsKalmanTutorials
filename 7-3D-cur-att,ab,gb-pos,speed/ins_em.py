import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ins_sig_gen import generate_signals
from ins_ekf import ins_ext_kfilter
import utils


# Config
imu_period = 0.01
accel_bias_std = 0.5
accel_w_std = 0.1
gyro_bias_std = np.deg2rad( 0.4 )
gyro_w_std = np.deg2rad( 0.1 )
imu_attitude0_std = np.deg2rad( 0.5 )
gnss_period = 0.25
gnss_speed_w_std = 0.1
gnss_dist_w_std = 1

'''
speed_changes = [
	[ +6, 5 ],
	[ +0, 60 ]
]
rot_changes_x = [
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    60 ],
]
rot_changes_y = [
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),  60 ],
]
rot_changes_z = [
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 5 * 360 ),   60 ],
]
'''

speed_changes = [
	[ +5,   5 ],
	[ 0,    5 ],
	[ 0,    5 ],
	[ 0,    5 ],
	[ 0,    5 ],
	[ 0,    5 ],
	[ 0,    5 ],
	[ 0,    5 ],
	[ 0,    5 ],
	[ 0,    5 ],
	[ 0,    0 ],
]
rot_changes_x = [
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( +40 ),  5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( -0 ),  5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),   5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    0 ],
]
rot_changes_y = [
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),   5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( -40 ),  5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),   5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    0 ],
]
rot_changes_z = [
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),   5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( -40 ),  5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),   5 ],
	[ np.deg2rad( 0 ),    0 ],
]


# Generate INS signals
[ imu_time, imu_accel, imu_gyro, 
  gnss_time, gnss_speed, gnss_dist,
  real_accel_bias, real_gyro_bias, 
  real_glob_attitude, real_glob_accel, real_glob_speed, real_glob_speed_norm, real_glob_dist 
] = generate_signals( 
	speed_changes, rot_changes_x, rot_changes_y, rot_changes_z,
	imu_period, accel_bias_std, accel_w_std, gyro_bias_std, gyro_w_std,
	gnss_period, gnss_speed_w_std, gnss_dist_w_std
)

# Initial params estimation
real_attitude0 = real_glob_attitude[0]
imu_attitude0 = real_attitude0 + np.matrix([
	# Psi error
	[ np.random.normal( 0, imu_attitude0_std ) ],
	# Theta error
	[ np.random.normal( 0, imu_attitude0_std ) ],
	# Gamma error
	[ np.random.normal( 0, imu_attitude0_std ) ]
])
print( 'Real psi0: '          + str( np.rad2deg( real_attitude0.item( ( 0, 0 ) ) ) ) + 
	   ', estimated psi0: '   + str( np.rad2deg( imu_attitude0.item(  ( 0, 0 ) ) ) ) )
print( 'Real theta0: '        + str( np.rad2deg( real_attitude0.item( ( 1, 0 ) ) ) ) + 
	   ', estimated theta0: ' + str( np.rad2deg( imu_attitude0.item(  ( 1, 0 ) ) ) ) )
print( 'Real gamma0: '        + str( np.rad2deg( real_attitude0.item( ( 2, 0 ) ) ) ) + 
	   ', estimated gamma0: ' + str( np.rad2deg( imu_attitude0.item(  ( 2, 0 ) ) ) ) )

# Estimate body motion
[ ins_state, ins_var ] = ins_ext_kfilter( 
	imu_time, imu_accel, imu_gyro, accel_bias_std, gyro_bias_std,
	imu_attitude0, imu_attitude0_std, 
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
'''
real_psi				= [ np.rad2deg( v.item( (0, 0) ) ) for v in real_glob_attitude ]
real_theta				= [ np.rad2deg( v.item( (1, 0) ) ) for v in real_glob_attitude ]
real_gamma				= [ np.rad2deg( v.item( (2, 0) ) ) for v in real_glob_attitude ]
'''

real_dcm 				= [ utils.get_dcm( v ) for v in real_glob_attitude ]
real_C11				= [ v.item( ( 0, 0 ) ) for v in real_dcm ]
real_C12				= [ v.item( ( 0, 1 ) ) for v in real_dcm ]
real_C13				= [ v.item( ( 0, 2 ) ) for v in real_dcm ]
real_C21				= [ v.item( ( 1, 0 ) ) for v in real_dcm ]
real_C22				= [ v.item( ( 1, 1 ) ) for v in real_dcm ]
real_C23				= [ v.item( ( 1, 2 ) ) for v in real_dcm ]
real_C31				= [ v.item( ( 2, 0 ) ) for v in real_dcm ]
real_C32				= [ v.item( ( 2, 1 ) ) for v in real_dcm ]
real_C33				= [ v.item( ( 2, 2 ) ) for v in real_dcm ]

# Estimated signals
ekf_glob_dist_x			= [ v.item( ( 0, 0 ) ) 	for v in ins_state ]
ekf_glob_dist_y			= [ v.item( ( 1, 0 ) ) 	for v in ins_state ]
ekf_glob_dist_z			= [ v.item( ( 2, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_x		= [ v.item( ( 3, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_y		= [ v.item( ( 4, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_z		= [ v.item( ( 5, 0 ) ) 	for v in ins_state ]
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
ekf_C11					= [ v.item( ( 12, 0 ) ) for v in ins_state ]
ekf_C11_var				= [ np.sqrt( v.item( ( 12, 12 ) ) ) for v in ins_var ]
ekf_C12					= [ v.item( ( 13, 0 ) ) for v in ins_state ]
ekf_C12_var				= [ np.sqrt( v.item( ( 13, 13 ) ) ) for v in ins_var ]
ekf_C13					= [ v.item( ( 14, 0 ) ) for v in ins_state ]
ekf_C13_var				= [ np.sqrt( v.item( ( 14, 14 ) ) ) for v in ins_var ]
ekf_C21					= [ v.item( ( 15, 0 ) ) for v in ins_state ]
ekf_C21_var				= [ np.sqrt( v.item( ( 15, 15 ) ) ) for v in ins_var ]
ekf_C22					= [ v.item( ( 16, 0 ) ) for v in ins_state ]
ekf_C22_var				= [ np.sqrt( v.item( ( 16, 16 ) ) ) for v in ins_var ]
ekf_C23					= [ v.item( ( 17, 0 ) ) for v in ins_state ]
ekf_C23_var				= [ np.sqrt( v.item( ( 17, 17 ) ) ) for v in ins_var ]
ekf_C31					= [ v.item( ( 18, 0 ) ) for v in ins_state ]
ekf_C31_var				= [ np.sqrt( v.item( ( 18, 18 ) ) ) for v in ins_var ]
ekf_C32					= [ v.item( ( 19, 0 ) ) for v in ins_state ]
ekf_C32_var				= [ np.sqrt( v.item( ( 19, 19 ) ) ) for v in ins_var ]
ekf_C33					= [ v.item( ( 20, 0 ) ) for v in ins_state ]
ekf_C33_var				= [ np.sqrt( v.item( ( 20, 20 ) ) ) for v in ins_var ]

'''
ekf_psi   = []
ekf_theta = []
ekf_gamma = []

for v in ins_state:
	dcm = np.matrix([
		[ v.item( ( 12, 0 ) ), v.item( ( 13, 0 ) ), v.item( ( 14, 0 ) ) ],
		[ v.item( ( 15, 0 ) ), v.item( ( 16, 0 ) ), v.item( ( 17, 0 ) ) ],
		[ v.item( ( 18, 0 ) ), v.item( ( 19, 0 ) ), v.item( ( 20, 0 ) ) ]
	])
	euler = utils.get_euler( dcm )
	ekf_psi.append(   np.rad2deg( euler.item( ( 0, 0 ) ) ) )
	ekf_theta.append( np.rad2deg( euler.item( ( 1, 0 ) ) ) )
	ekf_gamma.append( np.rad2deg( euler.item( ( 2, 0 ) ) ) )
'''

plt.figure()
plt.suptitle('Angular speed (IMU)')
plt.subplot(311)
plt.plot(imu_time, imu_rot_speed_x, 'r')
plt.subplot(312)
plt.plot(imu_time, imu_rot_speed_y, 'r')
plt.subplot(313)
plt.plot(imu_time, imu_rot_speed_z, 'r')

plt.figure()
plt.suptitle('Accel (IMU)')
plt.subplot(311)
plt.plot(imu_time, imu_accel_x, 'r')
plt.subplot(312)
plt.plot(imu_time, imu_accel_y, 'r')
plt.subplot(313)
plt.plot(imu_time, imu_accel_z, 'r')

plt.figure()
plt.suptitle('Accel bias')
plt.subplot(231)
plt.plot(imu_time, real_accel_bias_x, 'b', imu_time, ekf_accel_bias_x, 'c')
plt.subplot(232)
plt.plot(imu_time, real_accel_bias_y, 'b', imu_time, ekf_accel_bias_y, 'c')
plt.subplot(233)
plt.plot(imu_time, real_accel_bias_z, 'b', imu_time, ekf_accel_bias_z, 'c')
plt.subplot(234)
plt.plot(imu_time, ekf_accel_bias_x_var, 'c')
plt.subplot(235)
plt.plot(imu_time, ekf_accel_bias_y_var, 'c')
plt.subplot(236)
plt.plot(imu_time, ekf_accel_bias_z_var, 'c')

plt.figure()
plt.suptitle('Gyro bias')
plt.subplot(231)
plt.plot(imu_time, real_gyro_bias_x, 'b', imu_time, ekf_gyro_bias_x, 'c')
plt.subplot(232)
plt.plot(imu_time, real_gyro_bias_y, 'b', imu_time, ekf_gyro_bias_y, 'c')
plt.subplot(233)
plt.plot(imu_time, real_gyro_bias_z, 'b', imu_time, ekf_gyro_bias_z, 'c')
plt.subplot(234)
plt.plot(imu_time, ekf_gyro_bias_x_var, 'c')
plt.subplot(235)
plt.plot(imu_time, ekf_gyro_bias_y_var, 'c')
plt.subplot(236)
plt.plot(imu_time, ekf_gyro_bias_z_var, 'c')


plt.figure()
plt.suptitle('Speed norm')
plt.plot(imu_time, real_glob_speed_norm, 'b', gnss_time, gnss_speed_norm, 'g', imu_time, ekf_glob_speed_norm, 'c')

plt.figure()
plt.suptitle('Speed')
plt.subplot(311)
plt.plot(imu_time, real_glob_speed_x, 'b', imu_time, ekf_glob_speed_x, 'c')
plt.subplot(312)
plt.plot(imu_time, real_glob_speed_y, 'b', imu_time, ekf_glob_speed_y, 'c')
plt.subplot(313)
plt.plot(imu_time, real_glob_speed_z, 'b', imu_time, ekf_glob_speed_z, 'c')

plt.figure()
plt.suptitle('Position')
plt.subplot(311)
plt.plot(imu_time, real_glob_dist_x, 'b', gnss_time, gnss_dist_x, 'g', imu_time, ekf_glob_dist_x, 'c')
plt.subplot(312)
plt.plot(imu_time, real_glob_dist_y, 'b', gnss_time, gnss_dist_y, 'g', imu_time, ekf_glob_dist_y, 'c')
plt.subplot(313)
plt.plot(imu_time, real_glob_dist_z, 'b', gnss_time, gnss_dist_z, 'g', imu_time, ekf_glob_dist_z, 'c')

fig = plt.figure()
plt.suptitle('3D map')
ax = fig.add_subplot(111, projection='3d')
# Y is vertical for IMU
plt.axis('equal')
plt.plot(real_glob_dist_x, real_glob_dist_z, real_glob_dist_y, 'b')
plt.axis('equal')
plt.plot(gnss_dist_x, gnss_dist_z, gnss_dist_y, 'g')
plt.axis('equal')
plt.plot(ekf_glob_dist_x, ekf_glob_dist_z, ekf_glob_dist_y, 'c')

plt.figure()
plt.suptitle('Attitude')
'''
plt.subplot(311)
plt.plot(imu_time, real_psi, 'b', imu_time, ekf_psi, 'c')
plt.subplot(312)
plt.plot(imu_time, real_theta, 'b', imu_time, ekf_theta, 'c')
plt.subplot(313)
plt.plot(imu_time, real_gamma, 'b', imu_time, ekf_gamma, 'c')
'''
plt.subplot(331)
plt.plot(imu_time, real_C11, 'b', imu_time, ekf_C11, 'c')
plt.subplot(332)
plt.plot(imu_time, real_C12, 'b', imu_time, ekf_C12, 'c')
plt.subplot(333)
plt.plot(imu_time, real_C13, 'b', imu_time, ekf_C13, 'c')
plt.subplot(334)
plt.plot(imu_time, real_C21, 'b', imu_time, ekf_C21, 'c')
plt.subplot(335)
plt.plot(imu_time, real_C22, 'b', imu_time, ekf_C22, 'c')
plt.subplot(336)
plt.plot(imu_time, real_C23, 'b', imu_time, ekf_C23, 'c')
plt.subplot(337)
plt.plot(imu_time, real_C31, 'b', imu_time, ekf_C31, 'c')
plt.subplot(338)
plt.plot(imu_time, real_C32, 'b', imu_time, ekf_C32, 'c')
plt.subplot(339)
plt.plot(imu_time, real_C33, 'b', imu_time, ekf_C33, 'c')
plt.show()