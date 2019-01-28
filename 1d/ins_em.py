import numpy as np
import matplotlib.pyplot as plt
from ins_sig_gen import generate_signals
from ins_kalman import kfilter


# Config
acc_period = 0.01
acc_bias_const = 0.05
acc_bias_const_std = 0.1
acc_bias_w_std = 0.0000
acc_w_std = 0.03
gnss_period = 0.5
gnss_w_std = 1


# Generate INS signals
[time_acc, 
accel_real, speed_real, dist_real, 
accel_bias, accel_ins, speed_ins, dist_ins,
time_gnss, dist_gnss] = generate_signals( 
	acc_period, acc_w_std, acc_bias_const, acc_bias_w_std, 
	gnss_period, gnss_w_std 
)

# Kalman filter
[kf_bias, kf_speed_err, kf_dist_err] = kfilter(
	time_acc, dist_ins, time_gnss, dist_gnss, acc_bias_const_std, gnss_w_std
)

# Corrected motion data
speed_kf = [ins_val - err_val for ins_val, err_val in zip(speed_ins, kf_speed_err)]
dist_kf =  [ins_val - err_val for ins_val, err_val in zip(dist_ins, kf_dist_err)]
accel_kf = [ins_val - err_val for ins_val, err_val in zip(accel_ins, kf_bias)]

plt.figure()
plt.title('accel')
plt.plot(time_acc, accel_real, time_acc, accel_ins, time_acc, accel_kf, 'r')
plt.figure()
plt.title('speed')
plt.plot(time_acc, speed_real, time_acc, speed_ins, time_acc, speed_kf, 'r')
plt.figure()
plt.title('dst')
plt.plot(time_acc, dist_real, time_acc, dist_ins, time_gnss, dist_gnss, time_acc, dist_kf, 'r')
plt.figure()
plt.title('bias')
plt.plot(time_acc, accel_bias, time_acc, kf_bias, 'r')

plt.show()