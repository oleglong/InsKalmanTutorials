import numpy as np
import matplotlib.pyplot as plt
from ins_sig_gen import generate_signals


# Config
period = 0.01
noise_w_std = 0.1


[time, 
accel_real, speed_real, dist_real, 
accel_noisy, speed_noisy, dist_noisy] = generate_signals( period )

plt.figure()
plt.title('accel')
plt.plot(time, accel_real, time, accel_noisy)
plt.figure()
plt.title('speed')
plt.plot(time, speed_real, time, speed_noisy)
plt.figure()
plt.title('dst')
plt.plot(time, dist_real, time, dist_noisy)

plt.show()