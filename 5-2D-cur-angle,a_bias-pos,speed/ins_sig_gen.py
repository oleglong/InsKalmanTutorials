import numpy as np
import math
import matplotlib.pyplot as plt


def log_f( x, max, offset, scale ):
	return max / ( 1 + np.exp(-(x-offset)*scale) )
	
def param_f( x, max, duration ):
	offset = duration / 2
	# Magic
	scale = 2**( np.log2( 10 / duration ) + 1 )
	return log_f( x, max, offset, scale ) - log_f( 0, max, offset, scale )
	
def val_change_append( param_list, change, duration, period ):
	time = np.arange( 0, duration, period )
	min_val = param_list[-1]
	
	new_param_val = [ min_val + param_f( t, change, duration ) for t in time ]
	return param_list + new_param_val
	
def param_from_changes( change_list, period ):
	param_list = [ 0 ]
	for change in change_list:
		param_list = val_change_append( param_list, change[0], change[1], period )
	return param_list
	
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
	
def rot_speed_from_angle( angle, period ):
	rot_speed = []
	
	for s_prev, s in zip( angle[0 : -1], angle[1 :] ):		
		rot_speed.append(
			np.matrix([
				[ ( s.item( (0, 0) ) - s_prev.item( (0, 0) ) ) / period ]
			])
		)
		
	rot_speed.append(
		np.matrix([
			[ 0 ] 
		])
	)
	
	return rot_speed
	
def get_rot_matrix( alpha ):
	val = alpha.item( ( 0, 0 ) )
	
	return np.matrix ([
			[ np.cos( val ), -np.sin( val ) ],
			[ np.sin( val ),  np.cos( val ) ]
	])
	
def get_inv_rot_matrix( alpha ):
	val = alpha.item( ( 0, 0 ) )
	
	return np.matrix ([
			[  np.cos( val ), np.sin( val ) ],
			[ -np.sin( val ), np.cos( val ) ]
	])
		
def get_body_motion( global_alpha0, alpha_changes, speed_changes, period ):
	# Speed norm
	global_speed_norm = param_from_changes( speed_changes, period )
	# Tangential speed of body
	body_tang_speed = [
		np.matrix([
			# X
			[ speed ],
			# Y
			[ 0 ]
		])
		for speed in global_speed_norm
	]	
	# Convert to matrix 
	global_speed_norm = [
		np.matrix([
			[ speed ]
		])
		for speed in global_speed_norm
	]
	# Body rotation angle
	body_alpha = [
		np.matrix([
			[ alpha ]
		])
		for alpha in param_from_changes( alpha_changes, period )
	]	
	# Total IMU samples count
	if ( len( body_tang_speed ) != len( body_alpha ) ):
		print( "Angle changes time != speed changes time." )
			
	# Body rotation speed
	body_alpha_speed = rot_speed_from_angle( body_alpha, period )
	# Rotation angle from body frame to global frame
	global_alpha = [ global_alpha0 + alpha for alpha in body_alpha ]
	# Speed in global frame
	global_speed = []
	
	for alpha_g, speed_tang in zip( global_alpha, body_tang_speed ):
		global_speed.append( 
			get_rot_matrix( alpha_g ) * speed_tang
		)
	
	# Acceleration in global frame
	global_accel = accel_from_speed( global_speed, period )
	# Distance in global frame
	global_dist = dist_from_speed( global_speed, period )
	
	# Acceleration in body frame
	body_accel = []
	
	for alpha_g, accel_glob in zip( global_alpha, global_accel ):
		body_accel.append(
			get_inv_rot_matrix( alpha_g ) * accel_glob
		)
	
	return [ 
			body_alpha_speed, body_accel,
			global_alpha, global_accel, global_speed, global_speed_norm, global_dist 
	]
	
def get_gnss_signal( global_speed_norm, global_dist, gnss_speed_w_std, gnss_dist_w_std, gnss_period, imu_period ):
	gnss_time = np.arange( gnss_period, len( global_speed_norm ) * imu_period, gnss_period )
	gnss_dist = []
	gnss_speed = []
	gnss_coeff = 1 / imu_period
	
	for t in gnss_time:	
		# Distance
		dist_x = global_dist[int( gnss_coeff * t )].item( (0, 0) )
		dist_y = global_dist[int( gnss_coeff * t )].item( (1, 0) )
		# Distance white noise
		dist_noise_x = np.random.normal( 0, gnss_dist_w_std )
		dist_noise_y = np.random.normal( 0, gnss_dist_w_std )
		# GNSS distance
		gnss_dist.append( 
			np.matrix([
				# X
				[ dist_x + dist_noise_x ],
				# Y
				[ dist_y + dist_noise_y ]
			]) 
		)		
		
		# Speed
		speed_norm = global_speed_norm[int( gnss_coeff * t )]
		# Speed white noise
		speed_noise = np.matrix([
			[ np.random.normal( 0, gnss_speed_w_std ) ]
		])
		# GNSS speed
		gnss_speed.append( 
			speed_norm + speed_noise
		)	
	
	return [ gnss_time, gnss_speed, gnss_dist ]

def get_imu_signal( body_accel, body_alpha_speed, acc_bias_std, acc_w_std, gyro_w_std, period ):
	# Simulation time
	imu_time = np.arange( 0, len( body_accel ) )	
	imu_time = [ t * period for t in imu_time ]
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
	# Gyro white noise
	imu_gyro_w_noise = [
		np.matrix([
			# X
			[ np.random.normal(0, gyro_w_std) ],
			# Y
			[ np.random.normal(0, gyro_w_std) ]
		])
		for t in imu_time
	]
	# Accel bias
	imu_accel_bias_x = np.random.normal(0, acc_bias_std)
	imu_accel_bias_y = np.random.normal(0, acc_bias_std)
	imu_accel_bias = [
		np.matrix([
			# X
			[ imu_accel_bias_x ],
			# Y
			[ imu_accel_bias_y ]
		])
		for t in imu_time
	]
	# Noisy IMU accel output
	imu_accel = [ body + w_noise + bias for body, w_noise, bias in zip( body_accel, imu_accel_w_noise, imu_accel_bias ) ]
	# Noisy IMU gyro output
	imu_gyro = [ body + w_noise for body, w_noise in zip( body_alpha_speed, imu_gyro_w_noise ) ]
	
	return [ imu_time, imu_accel, imu_accel_bias, imu_gyro ]
	
def generate_signals(	
					 speed_changes,
					 alpha_changes,
					 
					 imu_period,
					 acc_bias_std,
					 acc_w_std,  
					 gyro_w_std,  
					 
					 gnss_period,
					 gnss_speed_w_std,
					 gnss_dist_w_std
					 ):
					 
	######################### REAL DATA
	# Body motion parameters. Reference data and ideal sensors output
	[ 
		body_alpha_speed, body_accel,
		global_alpha, global_accel, global_speed, global_speed_norm, global_dist 
	] =	get_body_motion( 
		# Initial alpha angle
		( np.random.rand() - 0.5 ) * np.pi, 
		alpha_changes, speed_changes, imu_period 
	)	
	######################### GNSS DATA
	[ 
		gnss_time, gnss_speed, gnss_dist 
	] = get_gnss_signal(
		global_speed_norm, global_dist, gnss_speed_w_std, gnss_dist_w_std, gnss_period, imu_period
	)	
	######################### IMU DATA
	[ 
		imu_time, imu_accel, imu_accel_bias, imu_gyro 
	] = get_imu_signal(
		body_accel, body_alpha_speed, acc_bias_std, acc_w_std, gyro_w_std, imu_period
	)

	return [ 
		# System inputs
		imu_time, imu_accel, imu_gyro,
		gnss_time, gnss_speed, gnss_dist,
		# Reference data
		imu_accel_bias, global_alpha,
		global_accel, global_speed, global_speed_norm, global_dist 
	]