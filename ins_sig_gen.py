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

def generate_signals(period, noise_std=0.02, max_dist=10, start_time=-15, duration=25, scale=1.3):
	end_time = start_time + duration

	time = np.arange(0, duration, period )
	accel = [ accel_f(t, start_time, max_dist, scale) for t in time ]
	[dist, speed] = get_motion_info( accel, period )
	
	# Generate noise
	noise_w = np.random.normal(0, noise_std, size=len(time))
	# Noisy accel
	accel_noisy = accel + noise_w
	# Noisy motion
	[dist_noisy, speed_noisy] = get_motion_info( accel_noisy, period )
	
	return [time, accel, speed, dist, accel_noisy, speed_noisy, dist_noisy]
	