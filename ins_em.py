import numpy as np
import matplotlib.pyplot as plt
from ins_sig_gen import generate_signals


# Config
acc_period = 0.01
acc_bias = 0.01
acc_w_std = 0.03
gnss_period = 1
gnss_w_std = 1


# Generate INS signals
[time_acc, 
accel_real, speed_real, dist_real, 
accel_noisy, speed_noisy, dist_noisy,
time_gnss, dist_gnss] = generate_signals( 
	acc_period, acc_bias, acc_w_std,
	gnss_period, gnss_w_std 
)

plt.figure()
plt.title('accel')
plt.plot(time_acc, accel_real, time_acc, accel_noisy)
plt.figure()
plt.title('speed')
plt.plot(time_acc, speed_real, time_acc, speed_noisy)
plt.figure()
plt.title('dst')
plt.plot(time_acc, dist_real, time_acc, dist_noisy, time_gnss, dist_gnss)

plt.show()