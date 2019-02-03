import numpy as np
import matplotlib.pyplot as plt


def log_f( x, max, offset, scale ):
	return max / ( 1 + np.exp(-(x-offset)*scale) )
	
def linear_speed_f( x, max, duration ):
	offset = duration / 2
	# Magic
	scale = 2**( np.log2( 10 / duration ) + 1 )
	return log_f( x, max, offset, scale ) - log_f( 0, max, offset, scale )
	
def linear_speed_append( speed_list, change, duration, period ):
	time = np.arange( 0, duration, period )
	min_val = speed_list[-1]
	
	new_speed = [ min_val + linear_speed_f( t, change, duration ) for t in time ]
	return speed_list + new_speed
	
def linear_speed_from_changes( logs, period ):
	speed_list = [ 0 ]
	for log in logs:
		speed_list = linear_speed_append( speed_list, log[0], log[1], period )
	return speed_list
	
def accel_from_speed( speed, period ):
	accel = []
	
	for s_prev, s in zip( speed[0 : -1], speed[1 :] ):		
		accel.append(
			np.matrix([
				# X
				[ ( s.item( (0, 0) ) - s_prev.item( (0, 0) ) ) / period ],
				# Y
				[ ( s.item( (1, 0) ) - s_prev.item( (1, 0) ) ) / period ],
			])
		)
		
	accel.append(
		np.matrix([
			# X
			[ 0 ],
			# Y
			[ 0 ],
		])
	)
	
	return accel
	
def speed_from_accel( accel, period ):
	speed = [
		np.matrix([
			# X
			[ 0 ],
			# Y
			[ 0 ],
		])
	]
	
	for a_prev in accel[0 : -1]:
		speed.append(
			np.matrix([
				# X
				[ speed[-1].item( (0, 0) ) + a_prev.item( (0, 0) ) * period ],
				# Y
				[ speed[-1].item( (1, 0) ) + a_prev.item( (1, 0) ) * period ],
			])
		)
		
	return speed
	
def dist_from_speed( speed, period ):
	dist = [
		np.matrix([
			# X
			[ 0 ],
			# Y
			[ 0 ],
		])
	]
	
	for s_prev, s in zip( speed[0 : -1], speed[1 : ] ):
		s_x_avg = ( s_prev.item( (0, 0) ) + s.item( (0, 0) ) ) / 2
		s_y_avg = ( s_prev.item( (1, 0) ) + s.item( (1, 0) ) ) / 2
		
		dist.append(
			np.matrix([
				# X
				[ dist[-1].item( (0, 0) ) + s_x_avg * period ],
				# Y
				[ dist[-1].item( (1, 0) ) + s_y_avg * period ],
			])
		)
	
	return dist
	
	

def generate_signals(	
					 speed_changes,
					 
					 imu_period,
					 acc_w_std,  
					 
					 gnss_period,
					 gnss_w_std
					):
	# Linear motion along one axis
	body_one_axis_speed = linear_speed_from_changes( speed_changes, imu_period )
	
	# Time data
	samples_count = len( body_one_axis_speed )
	max_time = samples_count * imu_period
	imu_time = np.arange( 0, max_time, imu_period )
	
	######################### BODY frame
	# Speed in axes of body
	body_speed = [ 
		np.matrix([
			# X
			[ body_one_axis_speed[ i ] ],
			# Y
			[ 0 ]
		])
		for i in range( samples_count )
	]
	# Accel in axes of body
	body_accel = accel_from_speed( body_speed, imu_period )
	# Initial orientation
	body_alpha0 = ( np.random.rand() - 0.5 ) * np.pi
	body_alpha = [
		np.matrix([
			[ body_alpha0 ]
		])
		for t in imu_time
	]	
	
	######################### GLOBAL frame
	# Rotation from body frame to global frame
	body_dcm0 = np.matrix ([
		[ np.cos( body_alpha0 ), -np.sin( body_alpha0 ) ],
		[ np.sin( body_alpha0 ), np.cos( body_alpha0 ) ]
	])
	global_accel = [ body_dcm0 * acc for acc in body_accel ]
	global_speed = speed_from_accel( global_accel, imu_period )
	global_dist = dist_from_speed( global_speed, imu_period )
	
	######################### GNSS
	gnss_time = np.arange( gnss_period, max_time, gnss_period )
	gnss_dist = []
	gnss_coeff = 1 / imu_period
	for t in gnss_time:
		gnss_dist.append( 
			np.matrix([
				# X
				[ global_dist[int( gnss_coeff * t )].item( (0, 0) ) + np.random.normal( 0, gnss_w_std ) ],
				# Y
				[ global_dist[int( gnss_coeff * t )].item( (1, 0) ) + np.random.normal( 0, gnss_w_std ) ]
			]) 
		)
	
	######################### IMU frame
	# Accel white noise
	imu_accel_w_noise = [
		np.matrix([
			# X
			[ np.random.normal(0, acc_w_std) ],
			# Y
			[ np.random.normal(0, acc_w_std) ]
		])
		for t in imu_time
	]
	# Noisy IMU accel output
	imu_accel = [ body + w_noise for body, w_noise in zip( body_accel, imu_accel_w_noise ) ]
	
	return [ # System inputs
			 imu_time, imu_accel, 
			 gnss_time, gnss_dist,
			 # Reference data
			 body_alpha,
			 global_accel, global_speed, global_dist ]