import numpy as np
import math
import matplotlib.pyplot as plt
import utils


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
				# Z
				[ ( s.item( (2, 0) ) - s_prev.item( (2, 0) ) ) / period ]
			])
		)
		
	accel.append(
		np.matrix([
			# X
			[ 0 ],
			# Y
			[ 0 ],
			# Z
			[ 0 ]
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
			# Z
			[ 0 ]
		])
	]
	
	for a_prev in accel[0 : -1]:
		speed.append(
			np.matrix([
				# X
				[ speed[-1].item( (0, 0) ) + a_prev.item( (0, 0) ) * period ],
				# Y
				[ speed[-1].item( (1, 0) ) + a_prev.item( (1, 0) ) * period ],
				# Z
				[ speed[-1].item( (2, 0) ) + a_prev.item( (2, 0) ) * period ],
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
			# Z
			[ 0 ]
		])
	]
	
	for s_prev, s in zip( speed[0 : -1], speed[1 : ] ):
		s_x_avg = ( s_prev.item( (0, 0) ) + s.item( (0, 0) ) ) / 2
		s_y_avg = ( s_prev.item( (1, 0) ) + s.item( (1, 0) ) ) / 2
		s_z_avg = ( s_prev.item( (2, 0) ) + s.item( (2, 0) ) ) / 2
		
		dist.append(
			np.matrix([
				# X
				[ dist[-1].item( (0, 0) ) + s_x_avg * period ],
				# Y
				[ dist[-1].item( (1, 0) ) + s_y_avg * period ],
				# Z
				[ dist[-1].item( (2, 0) ) + s_z_avg * period ]
			])
		)
	
	return dist
	
def rot_speed_from_angles( angles, period ):
	rot_speed = []
	
	for s_prev, s in zip( angles[0 : -1], angles[1 :] ):		
		rot_speed.append(
			np.matrix([
				# X
				[ ( s.item( (0, 0) ) - s_prev.item( (0, 0) ) ) / period ],
				# Y
				[ ( s.item( (1, 0) ) - s_prev.item( (1, 0) ) ) / period ],
				# Z
				[ ( s.item( (2, 0) ) - s_prev.item( (2, 0) ) ) / period ],
			])
		)
		
	rot_speed.append(
		np.matrix([
			# X
			[ 0 ],
			# Y
			[ 0 ],
			# Z
			[ 0 ]
		])
	)
	
	return rot_speed
		
# X - от хвоста к носу, Y - вверх, Z - от левого крыла к правому
def get_body_motion( psi0, theta0, gamma0, rot_changes_x, rot_changes_y, rot_changes_z, speed_changes, period ):
	# Speed norm
	global_speed_norm = [
		np.matrix([
			[ speed ]
		])
		for speed in param_from_changes( speed_changes, period )
	]
	
	# Body rotation angles
	body_attitude = []
	for x,y,z in zip( 
			param_from_changes( rot_changes_x, period ), 
			param_from_changes( rot_changes_y, period ), 
			param_from_changes( rot_changes_z, period ) ):
		
		body_attitude.append(
			np.matrix([
				[ x ],
				[ y ],
				[ z ]
			])
		)
	# Body rotation speed
	body_attitude_speed = rot_speed_from_angles( body_attitude, period )
	
	# Total IMU samples count
	if ( len( global_speed_norm ) != len( body_attitude_speed ) ):
		print( "Angle changes time != speed changes time." )
		
	# Global attitude in euler angles
	global_attitude_prev = np.matrix([
		# Psi - around Y(up direction) axis
		[ psi0 ],
		# Theta - around Z(right wing direction) axis
		[ theta0 ],
		# Gamma - around X(nose cone direction) axis
		[ gamma0 ]
	])

	global_attitude = []
	for body_w in body_attitude_speed:
		global_attitude_new = utils.attitude_euler_update( global_attitude_prev, body_w, period )
		global_attitude.append( global_attitude_new )
		global_attitude_prev = global_attitude_new	
			
	# Global speed
	global_speed = []
	for attitude, speed_norm in zip( global_attitude, global_speed_norm ):
		# Tangential speed of body
		speed_tang = np.matrix([
			# X
			[ speed_norm.item( ( 0, 0 ) ) ],
			# Y
			[ 0 ],
			# Z
			[ 0 ]
		])		
		global_speed.append( 
			utils.get_dcm( attitude ) * speed_tang
		)
		
	# Global acceleration
	global_accel = accel_from_speed( global_speed, period )
	global_seem_accel = [ 
		acc + np.matrix([
			# X
			[ 0 ],
			# Y
			[ 9.81 ],
			# Z
			[ 0 ],
		]) 
		for acc in global_accel
	]
	
	# Global distance
	global_dist = dist_from_speed( global_speed, period )
		
	# Acceleration in body frame
	body_accel = []	
	for attitude, accel in zip( global_attitude, global_seem_accel ):
		body_accel.append(
			utils.get_inv_dcm( attitude ) * accel
		)
	
	return [ 
			body_attitude_speed, body_accel,
			global_attitude, global_seem_accel, global_speed, global_speed_norm, global_dist 
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
		dist_z = global_dist[int( gnss_coeff * t )].item( (2, 0) )
		# Distance white noise
		dist_noise_x = np.random.normal( 0, gnss_dist_w_std )
		dist_noise_y = np.random.normal( 0, gnss_dist_w_std )
		dist_noise_z = np.random.normal( 0, gnss_dist_w_std )
		# GNSS distance
		gnss_dist.append( 
			np.matrix([
				# X
				[ dist_x + dist_noise_x ],
				# Y
				[ dist_y + dist_noise_y ],
				# Z
				[ dist_z + dist_noise_z ]
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

def get_imu_signal( body_accel, body_attitude_speed, accel_bias0, acc_w_std, gyro_bias0, gyro_w_std, period ):
	# Simulation time
	imu_time = np.arange( 0, len( body_accel ) )	
	imu_time = [ t * period for t in imu_time ]
	# Accel white noise
	imu_accel_w_noise = [
		np.matrix([
			# X
			[ np.random.normal(0, acc_w_std) ],
			# Y
			[ np.random.normal(0, acc_w_std) ],
			# Z
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
			[ np.random.normal(0, gyro_w_std) ],
			# Z
			[ np.random.normal(0, gyro_w_std) ]
		])
		for t in imu_time
	]
	# Accel bias
	imu_accel_bias = [
		np.matrix([
			[ accel_bias0.item( ( 0, 0 ) ) ],
			[ accel_bias0.item( ( 1, 0 ) ) ],
			[ accel_bias0.item( ( 2, 0 ) ) ]
		])
		for t in imu_time
	]
	# Gyro bias
	imu_gyro_bias = [
		np.matrix([
			[ gyro_bias0.item( ( 0, 0 ) ) ],
			[ gyro_bias0.item( ( 1, 0 ) ) ],
			[ gyro_bias0.item( ( 2, 0 ) ) ],
		])
		for t in imu_time
	]
	# Noisy IMU accel output
	imu_accel = [ body + w_noise + bias for body, w_noise, bias in zip( body_accel, imu_accel_w_noise, imu_accel_bias ) ]
	# Noisy IMU gyro output
	imu_gyro =  [ body + w_noise + bias for body, w_noise, bias in zip( body_attitude_speed, imu_gyro_w_noise, imu_gyro_bias ) ]
	
	return [ imu_time, imu_accel, imu_accel_bias, imu_gyro, imu_gyro_bias ]
	
def generate_signals(	
					 speed_changes,
					 rot_changes_x, 
					 rot_changes_y, 
					 rot_changes_z,
					 attitude0,
					 
					 imu_period,
					 acc_bias0,
					 acc_w_std, 
					 gyro_bias0,					 
					 gyro_w_std,  
					 
					 gnss_period,
					 gnss_speed_w_std,
					 gnss_dist_w_std
					 ):
					 
	######################### REAL DATA
	# Body motion parameters. Reference data and ideal sensors output
	[ 
		body_attitude_speed, body_accel,
		global_attitude, global_accel, global_speed, global_speed_norm, global_dist 
	] =	get_body_motion( 
		# Psi initial value
		attitude0.item( ( 0, 0 ) ), 
		# Theta initial value
		attitude0.item( ( 1, 0 ) ), 
		# Gamma initial value
		attitude0.item( ( 2, 0 ) ), 
		rot_changes_x, rot_changes_y, rot_changes_z,
		speed_changes, imu_period 
	)	
	######################### GNSS DATA
	[ 
		gnss_time, gnss_speed, gnss_dist 
	] = get_gnss_signal(
		global_speed_norm, global_dist, gnss_speed_w_std, gnss_dist_w_std, gnss_period, imu_period
	)	
	######################### IMU DATA
	[ 
		imu_time, imu_accel, imu_accel_bias, imu_gyro, imu_gyro_bias
	] = get_imu_signal(
		body_accel, body_attitude_speed, acc_bias0, acc_w_std, gyro_bias0, gyro_w_std, imu_period
	)

	return [ 
		# System inputs
		imu_time, imu_accel, imu_gyro,
		gnss_time, gnss_speed, gnss_dist,
		# Reference data
		imu_accel_bias, imu_gyro_bias, global_attitude,
		global_accel, global_speed, global_speed_norm, global_dist 
	]