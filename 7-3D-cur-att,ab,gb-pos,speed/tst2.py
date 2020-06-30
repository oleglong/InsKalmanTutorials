import numpy as np
import matplotlib.pyplot as plt
from ins_sig_gen import generate_signals
import utils


# Config
imu_period = 0.005
accel_bias_std = 0.0
accel_w_std = 0.0
gyro_bias_std = np.deg2rad( 1.0 )
gyro_w_std = np.deg2rad( 0.0 )
imu_attitude0_std = np.deg2rad( 0.0 )
gnss_period = 0.25
gnss_speed_w_std = 0.1
gnss_dist_w_std = 1

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
	
	[ np.deg2rad( +40 ),  5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	
	[ np.deg2rad( +40 ),  5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	
	[ np.deg2rad( 0 ),    0 ],
]
rot_changes_y = [
	[ np.deg2rad( 0 ),    5 ],
	
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( +40 ),  5 ],
	[ np.deg2rad( 0 ),    5 ],
	
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( +40 ),  5 ],
	[ np.deg2rad( 0 ),    5 ],
	
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( +40 ),  5 ],
	[ np.deg2rad( 0 ),    5 ],
	
	[ np.deg2rad( 0 ),    0 ],
]
rot_changes_z = [
	[ np.deg2rad( 0 ),    5 ],
	
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( +40 ),  5 ],
	
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( +40 ),  5 ],
	
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( 0 ),    5 ],
	[ np.deg2rad( +40 ),  5 ],
	
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

# Real signals
euler_psi	= [ np.rad2deg( v.item( ( 0, 0 ) ) ) for v in real_glob_attitude ]
euler_theta	= [ np.rad2deg( v.item( ( 1, 0 ) ) ) for v in real_glob_attitude ]
euler_gamma	= [ np.rad2deg( v.item( ( 2, 0 ) ) ) for v in real_glob_attitude ]

attitude_ins = utils.get_dcm( real_glob_attitude[ 0 ] )
attitude_err = np.matrix([ 
	[ 0, 0, 0],
	[ 0, 0, 0],
	[ 0, 0, 0]
 ])
dcm_psi   = [ euler_psi[ 0 ] ]
dcm_theta = [ euler_theta[ 0 ] ]
dcm_gamma = [ euler_gamma[ 0 ] ]

for gyro, gyro_bias in zip( imu_gyro[ 1 : ], real_gyro_bias[ 1 : ] ):
	wx = gyro.item( ( 0, 0 ) )
	wy = gyro.item( ( 1, 0 ) )
	wz = gyro.item( ( 2, 0 ) )
	
	wx_b = gyro_bias.item( ( 0, 0 ) )
	wy_b = gyro_bias.item( ( 1, 0 ) )
	wz_b = gyro_bias.item( ( 2, 0 ) )
	
	Wb = np.matrix([
		[ 0,     -wz_b, wy_b  ],
		[ wz_b,  0,     -wx_b ],
		[ -wy_b, wx_b,  0     ]
	])
	Wi = np.matrix([
		[ 0,   -wz,  wy ],
		[ wz,  0,   -wx ],
		[ -wy, wx,  0   ]
	])
	E = np.matrix([
		[ 1, 0, 0 ],
		[ 0, 1, 0 ],
		[ 0, 0, 1 ]
	])
	
	attitude_err = ( attitude_ins * (-Wb) * imu_period ) + ( attitude_err + attitude_err * ( Wi - Wb ) * imu_period )
	attitude_ins = attitude_ins + attitude_ins * Wi * imu_period
	
	attitude_dcm = attitude_ins + attitude_err
	atitude_euler = utils.get_euler( attitude_dcm )
	
	dcm_psi.append(   np.rad2deg( atitude_euler.item( ( 0, 0 ) ) ) )
	dcm_theta.append( np.rad2deg( atitude_euler.item( ( 1, 0 ) ) ) )
	dcm_gamma.append( np.rad2deg( atitude_euler.item( ( 2, 0 ) ) ) )
	

plt.figure()
plt.suptitle('Attitude')
plt.subplot(311)
plt.plot(imu_time, euler_psi, 'b', imu_time, dcm_psi, 'c')
plt.subplot(312)
plt.plot(imu_time, euler_theta, 'b', imu_time, dcm_theta, 'c')
plt.subplot(313)
plt.plot(imu_time, euler_gamma, 'b', imu_time, dcm_gamma, 'c')

plt.show()