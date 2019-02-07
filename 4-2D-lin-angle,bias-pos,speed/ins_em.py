import numpy as np
import matplotlib.pyplot as plt
from ins_sig_gen import generate_signals
from ins_ekf import ins_ext_kfilter


# Config
imu_period = 0.01
accel_bias_std = 0.3
accel_w_std = 0.05
imu_body_alpha0_std = 3 / 180 * np.pi
gnss_period = 0.5
gnss_speed_w_std = 0.7
gnss_dist_w_std = 5
speed_changes = [
	[ +3, 10 ],
	[  0, 10 ],
	[ +4, 12 ],
	[ -2, 15 ],
	[  0, 30 ],
	[ +5, 8  ],
	[ -4, 15 ],
	[  0, 25 ],
	[ -4, 10 ],
	[ -2, 15 ]
]

# Generate INS signals
[ imu_time, imu_accel, 
  gnss_time, gnss_speed, gnss_dist,
  real_accel_bias, real_body_angles, 
  real_glob_accel, real_glob_speed, real_glob_speed_norm, real_glob_dist 
] = generate_signals( 
	speed_changes,
	imu_period, accel_bias_std, accel_w_std,
	gnss_period, gnss_speed_w_std, gnss_dist_w_std
)
# Initial params estimation
real_body_alpha0 = real_body_angles[0].item( ( 0, 0 ) )
imu_body_alpha0 = real_body_alpha0 + np.random.normal( 0, imu_body_alpha0_std )
print('Real a0: ' + str(real_body_alpha0 / np.pi * 180) + ', estimated a0: ' + str(imu_body_alpha0 / np.pi * 180) )

# Estimate body motion
[ ins_state, ins_var ] = ins_ext_kfilter( 
	imu_time, imu_accel, accel_bias_std, 
	imu_body_alpha0, imu_body_alpha0_std, 
	gnss_time, gnss_speed, gnss_dist, gnss_speed_w_std, gnss_dist_w_std )


imu_accel_x 		= [ v.item( (0, 0) ) for v in imu_accel ]
imu_accel_y 		= [ v.item( (1, 0) ) for v in imu_accel ]
gnss_dist_x 		= [ v.item( (0, 0) ) for v in gnss_dist ]
gnss_dist_y 		= [ v.item( (1, 0) ) for v in gnss_dist ]
gnss_speed_norm		= [ v.item( (0, 0) ) for v in gnss_speed ]

real_accel_bias_x 	= [ v.item( (0, 0) ) for v in real_accel_bias ]
real_accel_bias_y 	= [ v.item( (1, 0) ) for v in real_accel_bias ]
real_glob_accel_x 	= [ v.item( (0, 0) ) for v in real_glob_accel ]
real_glob_accel_y 	= [ v.item( (1, 0) ) for v in real_glob_accel ]
real_glob_speed_x 	= [ v.item( (0, 0) ) for v in real_glob_speed ]
real_glob_speed_y 	= [ v.item( (1, 0) ) for v in real_glob_speed ]
real_glob_speed_norm = [ v.item( (0, 0) ) for v in real_glob_speed_norm ]
real_glob_dist_x 	= [ v.item( (0, 0) ) for v in real_glob_dist ]
real_glob_dist_y 	= [ v.item( (1, 0) ) for v in real_glob_dist ]
real_body_alpha		= [ v.item( (0, 0) ) / np.pi * 180 for v in real_body_angles ]

ekf_glob_dist_x			= [ v.item( ( 0, 0 ) ) 	for v in ins_state ]
ekf_glob_dist_y			= [ v.item( ( 1, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_x		= [ v.item( ( 2, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_y		= [ v.item( ( 3, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_norm		= [ np.sqrt( x**2 + y**2 ) for x,y in zip( ekf_glob_speed_x, ekf_glob_speed_y ) ]
ekf_accel_bias_x		= [ v.item( ( 4, 0 ) ) 	for v in ins_state ]
ekf_accel_bias_x_var	= [ np.sqrt( v.item( ( 4, 4 ) ) ) 	for v in ins_var ]
ekf_accel_bias_y		= [ v.item( ( 5, 0 ) ) 	for v in ins_state ]
ekf_accel_bias_y_var	= [ np.sqrt( v.item( ( 5, 5 ) ) ) 	for v in ins_var ]
ekf_alpha				= [ v.item( ( 6, 0 ) ) / np.pi * 180 for v in ins_state ]
ekf_alpha_var			= [ np.sqrt( v.item( ( 6, 6 ) ) ) / np.pi * 180 for v in ins_var ]

'''
plt.figure()
plt.title('accel')
plt.subplot(211)
plt.plot(imu_time, imu_accel_x, 'r', imu_time, real_glob_accel_x, 'b')
plt.subplot(212)
plt.plot(imu_time, imu_accel_y, 'r', imu_time, real_glob_accel_y, 'b')
'''

plt.figure()
plt.title('bias')
plt.subplot(221)
plt.plot(imu_time, real_accel_bias_x, 'b', imu_time, ekf_accel_bias_x, 'c')
plt.subplot(222)
plt.plot(imu_time, real_accel_bias_y, 'b', imu_time, ekf_accel_bias_y, 'c')
plt.subplot(223)
plt.plot(imu_time, ekf_accel_bias_x_var, 'c')
plt.subplot(224)
plt.plot(imu_time, ekf_accel_bias_y_var, 'c')

plt.figure()
plt.title('speed norm')
plt.plot(imu_time, real_glob_speed_norm, 'b', gnss_time, gnss_speed_norm, 'g', imu_time, ekf_glob_speed_norm, 'c')

plt.figure()
plt.title('speed')
plt.subplot(211)
plt.plot(imu_time, real_glob_speed_x, 'b', imu_time, ekf_glob_speed_x, 'c')
plt.subplot(212)
plt.plot(imu_time, real_glob_speed_y, 'b', imu_time, ekf_glob_speed_y, 'c')

plt.figure()
plt.title('dist')
plt.subplot(211)
plt.plot(imu_time, real_glob_dist_x, 'b', gnss_time, gnss_dist_x, 'g', imu_time, ekf_glob_dist_x, 'c')
plt.subplot(212)
plt.plot(imu_time, real_glob_dist_y, 'b', gnss_time, gnss_dist_y, 'g', imu_time, ekf_glob_dist_y, 'c')

plt.figure()
plt.title('surf')
plt.plot(real_glob_dist_x, real_glob_dist_y, 'b', gnss_dist_x, gnss_dist_y, 'g', ekf_glob_dist_x, ekf_glob_dist_y, 'c')

plt.figure()
plt.title('alpha')
plt.subplot(211)
plt.plot(imu_time, real_body_alpha, 'b', imu_time, ekf_alpha, 'c' )
plt.subplot(212)
plt.title('alpha var')
plt.plot(imu_time, ekf_alpha_var, 'c' )

plt.show()