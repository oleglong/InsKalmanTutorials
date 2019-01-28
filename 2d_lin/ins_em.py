import numpy as np
import matplotlib.pyplot as plt
from ins_sig_gen import generate_signals
#from ins_kalman import kfilter


# Config
imu_period = 0.01
acc_bias_std = 0.1
acc_w_std = 0.03
gnss_period = 0.5
gnss_w_std = 5
body_alpha = np.pi / 4 + 0.1
speed_changes = [
	[10, 5],
	[-5, 7],
	[0, 10]
]

# Generate INS signals
[time, 
 imu_accel, imu_accel_bias,
 global_accel, global_speed, global_dist,
 gnss_time, gnss_dist] = generate_signals( 
	speed_changes,
	imu_period, acc_bias_std, acc_w_std,
	gnss_period, gnss_w_std,
	body_alpha
)

imu_accel_x 		= [ v.item( (0, 0) ) for v in imu_accel ]
imu_accel_y 		= [ v.item( (1, 0) ) for v in imu_accel ]
imu_accel_bias_x 	= [ v.item( (0, 0) ) for v in imu_accel_bias ]
imu_accel_bias_y 	= [ v.item( (1, 0) ) for v in imu_accel_bias ]
global_accel_x 		= [ v.item( (0, 0) ) for v in global_accel ]
global_accel_y 		= [ v.item( (1, 0) ) for v in global_accel ]
global_speed_x 		= [ v.item( (0, 0) ) for v in global_speed ]
global_speed_y 		= [ v.item( (1, 0) ) for v in global_speed ]
global_dist_x 		= [ v.item( (0, 0) ) for v in global_dist ]
global_dist_y 		= [ v.item( (1, 0) ) for v in global_dist ]
gnss_dist_x 		= [ v.item( (0, 0) ) for v in gnss_dist ]
gnss_dist_y 		= [ v.item( (1, 0) ) for v in gnss_dist ]

plt.figure()
plt.title('accel')
plt.subplot(211)
plt.plot(time, imu_accel_x, 'r', time, imu_accel_bias_x, 'm', time, global_accel_x, 'b')
plt.subplot(212)
plt.plot(time, imu_accel_y, 'r', time, imu_accel_bias_y, 'm', time, global_accel_y, 'b')

plt.figure()
plt.title('speed')
plt.subplot(211)
plt.plot(time, global_speed_x, 'b')
plt.subplot(212)
plt.plot(time, global_speed_y, 'b')

plt.figure()
plt.title('dist')
plt.subplot(211)
plt.plot(time, global_dist_x, 'b', gnss_time, gnss_dist_x, 'g')
plt.subplot(212)
plt.plot(time, global_dist_y, 'b', gnss_time, gnss_dist_y, 'g')

plt.show()