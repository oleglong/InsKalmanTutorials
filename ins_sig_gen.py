import numpy as np
import matplotlib.pyplot as plt


def log_f( x, max_val ):
	return max_val / ( 1 + np.exp(-x-x0) )
	
def log_der2_f( x, max_val ):
	val1 = np.exp(-x) + 1
	return max_val * ( -np.exp(-x) * val1 + 2 * np.exp( -2 * x ) ) / val1 ** 3
	
def accel_f( t, t0, dst_max, scale=1 ):
	x = ( t + t0 ) / scale
	x0 = t0 / scale
	return log_der2_f( x, dst_max ) - log_der2_f( x0, dst_max )
	
def get_motion_info( accel_data, period ):
	speed = [ 0 ]
	dist = [ 0 ]
	
	for accel_prev, accel_new in zip( accel_data[0 : -1], accel_data[1 :] ):
		accel = ( accel_new + accel_prev ) / 2
		
		speed_prev = speed[-1]
		dist_prev = dist[-1]
		
		speed.append( speed_prev + accel * period )
		dist.append( dist_prev + speed_prev * period + accel * period ** 2 / 2 )
		
	return [ dist, speed ]

def generate_signals(	
					 imu_period, 
					 acc_bias, 
					 acc_w_std, 
					 
					 gnss_period,
					 gnss_w_std,
						
					 max_dist=10, 
					 start_time=-15, 
					 duration=25, 
					 scale=1.3
					):
					
	end_time = start_time + duration

	# Real signals
	time = np.arange(0, duration, imu_period)
	accel = [ accel_f(t, start_time, max_dist, scale) for t in time ]
	[dist, speed] = get_motion_info(accel, imu_period)
	
	# IMU signals
	accel_noise = np.random.normal(0, acc_w_std, size=len(time))
	accel_noisy = accel + accel_noise + acc_bias
	[dist_noisy, speed_noisy] = get_motion_info(accel_noisy, imu_period)
	
	# Gnss signal
	time_gnss = np.arange(gnss_period, duration, gnss_period)
	gnss_dist = []
	coeff = gnss_period / imu_period
	for t in time_gnss:
		gnss_dist.append(dist[int(coeff * t)])
		
	gnss_dist = gnss_dist + np.random.normal(0, gnss_w_std, size=len(time_gnss))
	
	return [time, accel, speed, dist, accel_noisy, speed_noisy, dist_noisy, time_gnss, gnss_dist]
	