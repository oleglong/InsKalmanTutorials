import numpy as np
import matplotlib.pyplot as plt
from ins_sig_gen import generate_signals
from ins_ekf import ins_ext_kfilter


# Config
imu_period = 0.01
accel_w_std = 0.03
gnss_period = 0.5
gnss_w_std = 5
imu_body_alpha0_std = 5 / 180 * np.pi
speed_changes = [
	[ +3, 10 ],
	[  0, 10 ],
	[ +4, 12 ],
	[ -2, 15 ],
	[ +5, 8  ],
	[ -4, 15 ],
	[ -4, 10 ],
	[ -2, 15 ]
]

# Generate INS signals
[ imu_time, imu_accel, 
  gnss_time, gnss_dist,
  real_body_angles, 
  real_glob_accel, real_glob_speed, real_glob_dist 
] = generate_signals( 
	speed_changes,
	imu_period, accel_w_std,
	gnss_period, gnss_w_std
)
# Initial params estimation
real_body_alpha0 = real_body_angles[0].item( ( 0, 0 ) )
imu_body_alpha0 = real_body_alpha0 + np.random.normal( 0, imu_body_alpha0_std )
print('Real a0: ' + str(real_body_alpha0 / np.pi * 180) + ', estimated a0: ' + str(imu_body_alpha0 / np.pi * 180) )

# Estimate body motion
ins_state = ins_ext_kfilter( 
	imu_time, imu_accel, imu_body_alpha0, imu_body_alpha0_std, 
	gnss_time, gnss_dist, gnss_w_std )


imu_accel_x 		= [ v.item( (0, 0) ) for v in imu_accel ]
imu_accel_y 		= [ v.item( (1, 0) ) for v in imu_accel ]
gnss_dist_x 		= [ v.item( (0, 0) ) for v in gnss_dist ]
gnss_dist_y 		= [ v.item( (1, 0) ) for v in gnss_dist ]

real_glob_accel_x 	= [ v.item( (0, 0) ) for v in real_glob_accel ]
real_glob_accel_y 	= [ v.item( (1, 0) ) for v in real_glob_accel ]
real_glob_speed_x 	= [ v.item( (0, 0) ) for v in real_glob_speed ]
real_glob_speed_y 	= [ v.item( (1, 0) ) for v in real_glob_speed ]
real_glob_dist_x 	= [ v.item( (0, 0) ) for v in real_glob_dist ]
real_glob_dist_y 	= [ v.item( (1, 0) ) for v in real_glob_dist ]
real_body_alpha		= [ v.item( (0, 0) ) / np.pi * 180 for v in real_body_angles ]

ekf_glob_dist_x		= [ v.item( ( 0, 0 ) ) 	for v in ins_state ]
ekf_glob_dist_y		= [ v.item( ( 1, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_x	= [ v.item( ( 2, 0 ) ) 	for v in ins_state ]
ekf_glob_speed_y	= [ v.item( ( 3, 0 ) ) 	for v in ins_state ]
ekf_alpha			= [ v.item( ( 4, 0 ) ) / np.pi * 180 for v in ins_state ]

plt.figure()
plt.title('accel')
plt.subplot(211)
plt.plot(imu_time, imu_accel_x, 'r', imu_time, real_glob_accel_x, 'b')
plt.subplot(212)
plt.plot(imu_time, imu_accel_y, 'r', imu_time, real_glob_accel_y, 'b')

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
plt.plot(imu_time, real_body_alpha, 'b', imu_time, ekf_alpha, 'c' )

plt.show()