import numpy as np
import numpy.ma as ma

# State prediction function
def exec_f_func( x_vect, u_vect, period ):
	pos_gx   	 	 = x_vect.item( ( 0, 0 ) )
	pos_gy   	 	 = x_vect.item( ( 1, 0 ) )
	speed_gx	 	 = x_vect.item( ( 2, 0 ) )
	speed_gy 	 	 = x_vect.item( ( 3, 0 ) )
	accel_bias_x 	 = x_vect.item( ( 4, 0 ) )
	accel_bias_y 	 = x_vect.item( ( 5, 0 ) )
	alpha    	 	 = x_vect.item( ( 6, 0 ) )
	alpha_speed_bias = x_vect.item( ( 7, 0 ) )
	
	est_accel_ix = u_vect.item( ( 0, 0 ) ) - accel_bias_x
	est_accel_iy = u_vect.item( ( 1, 0 ) ) - accel_bias_y
	est_alpha_speed = u_vect.item( ( 2, 0 ) ) - alpha_speed_bias
	
	accel_gx = np.cos( alpha ) * est_accel_ix - np.sin( alpha ) * est_accel_iy
	accel_gy = np.sin( alpha ) * est_accel_ix + np.cos( alpha ) * est_accel_iy
			
	dt2 = 0.5 * period**2	
	dt = period
	
	return np.matrix([
		[ pos_gx + speed_gx * dt + accel_gx * dt2 ],
		[ pos_gy + speed_gy * dt + accel_gy * dt2 ],
		[ speed_gx + accel_gx * dt ],
		[ speed_gy + accel_gy * dt ],
		[ accel_bias_x ],
		[ accel_bias_y ],
		[ alpha + est_alpha_speed * dt],
		[ alpha_speed_bias ]
	])
	
# State prediction Jacobian matrix
def get_F_matrix( x_vect, u_vect, period ):
	accel_bias_x = x_vect.item( ( 4, 0 ) )
	accel_bias_y = x_vect.item( ( 5, 0 ) )
	alpha = x_vect.item( ( 6, 0 ) )
	
	est_accel_ix = u_vect.item( ( 0, 0 ) ) - accel_bias_x
	est_accel_iy = u_vect.item( ( 1, 0 ) ) - accel_bias_y
	alpha_speed = u_vect.item( ( 2, 0 ) )
	
	# d(accel_gx)/d(accel_bias_x)
	d_agx_d_bx = -np.cos( alpha )
	# d(accel_gy)/d(accel_bias_x)
	d_agy_d_bx = -np.sin( alpha )
	# d(accel_gx)/d(accel_bias_y)
	d_agx_d_by = np.sin( alpha )
	# d(accel_gy)/d(accel_bias_y)
	d_agy_d_by = -np.cos( alpha )	
	# d(accel_gx)/d(alpha)
	d_agx_d_a = -est_accel_ix * np.sin( alpha ) - est_accel_iy * np.cos( alpha )
	# d(accel_gy)/d(alpha)
	d_agy_d_a =  est_accel_ix * np.cos( alpha ) - est_accel_iy * np.sin( alpha )
		
	dt2 = 0.5 * period**2	
	dt = period
	
	F = np.matrix([
		[1,   0,   dt,   0,   d_agx_d_bx * dt2,  d_agx_d_by * dt2,  d_agx_d_a * dt2,  0   ],
		[0,   1,   0,    dt,  d_agy_d_bx * dt2,  d_agy_d_by * dt2,  d_agy_d_a * dt2,  0   ],
		[0,   0,   1,    0,   d_agx_d_bx * dt,   d_agx_d_by * dt,   d_agx_d_a * dt,   0   ],
		[0,   0,   0,    1,   d_agy_d_bx * dt,   d_agy_d_by * dt,   d_agy_d_a * dt,   0   ],
		[0,   0,   0,    0,   1,                 0,                 0,    			  0   ],
		[0,   0,   0,    0,   0,                 1,                 0,    			  0   ],
		[0,   0,   0,    0,   0,                 0,                 1,    		      -dt ],
		[0,   0,   0,    0,   0,                 0,                 0,    			  1   ]
	])
	
	return F
	
# Measurement function
def exec_h_func( x_vect, period ):
	pos_gx   = x_vect.item( ( 0, 0 ) )
	pos_gy   = x_vect.item( ( 1, 0 ) )
	speed_gx = x_vect.item( ( 2, 0 ) )
	speed_gy = x_vect.item( ( 3, 0 ) )
	speed_norm = np.sqrt( speed_gx**2 + speed_gy**2 )
	
	return np.matrix([
		[ pos_gx ],
		[ pos_gy ],
		[ speed_norm ]
	])	
	
# Measurement Jacobian matrix
def get_H_matrix( x_vect, period ):
	speed_gx = x_vect.item( ( 2, 0 ) )
	speed_gy = x_vect.item( ( 3, 0 ) )
	
	# d(speed_norm)/d(speed_gx)
	d_sn_d_sgx = speed_gx / np.sqrt( speed_gx**2 + speed_gy**2 )
	d_sn_d_sgy = speed_gy / np.sqrt( speed_gx**2 + speed_gy**2 )

	return np.matrix([
		[ 1,  0,  0,          0,           0,  0,  0, 0 ],
		[ 0,  1,  0,          0,           0,  0,  0, 0 ],
		[ 0,  0,  d_sn_d_sgx, d_sn_d_sgy,  0,  0,  0, 0 ]
	])
	
def ins_ext_kfilter( imu_time, imu_accel, imu_gyro, accel_bias_std, gyro_bias_std,
					 alpha0, alpha0_std, 
					 gnss_time, gnss_speed, gnss_dist, gnss_speed_std, gnss_dist_std ):
	# Output data
	state_list = []
	var_list = []
	
	# IMU sampling period
	imu_dt = imu_time[1] - imu_time[0]
	# State matrix
	X = np.matrix([ 
		# X position
		[0.0],
		# Y position
		[0.0],
		# X speed
		[0.0],
		# Y speed
		[0.0],
		# Accel X bias
		[0.0],
		# Accel Y bias
		[0.0],
		# alpha
		[alpha0],
		# Gyro alpha speed bias
		[0.0]
	])
	# Process noise matrix
	Q = np.matrix([
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0]
	])
	# Measurement noise matrix
	R = np.matrix([
		[gnss_dist_std**2,  0,                 0,                  ],
		[0,                 gnss_dist_std**2,  0,                  ],
		[0,                 0,                 gnss_speed_std**2,  ]
	])
	# State covariance matrix
	P = np.matrix([
		[0, 0, 0, 0, 0,                 0,                 0,              0                ],
		[0, 0, 0, 0, 0,                 0,                 0,              0                ],
		[0, 0, 0, 0, 0,                 0,                 0,              0                ],
		[0, 0, 0, 0, 0,                 0,                 0,              0                ],
		[0, 0, 0, 0, accel_bias_std**2, 0,                 0,              0                ],
		[0, 0, 0, 0, 0,                 accel_bias_std**2, 0,              0                ],
		[0, 0, 0, 0, 0,                 0,                 alpha0_std**2,  0                ],
		[0, 0, 0, 0, 0,                 0,                 0, 	           gyro_bias_std**2 ],
	])
	
	gnss_i = 0
	for t, accel, gyro in zip( imu_time, imu_accel, imu_gyro ):
		# Gnss data available
		if ( gnss_i < len( gnss_time ) and t > gnss_time[ gnss_i ] ):			
			# ----- Kalman update step
			H = get_H_matrix( X, imu_dt )
			gnss_dist_x = gnss_dist[ gnss_i ].item( ( 0, 0 ) )
			gnss_dist_y = gnss_dist[ gnss_i ].item( ( 1, 0 ) )
			gnss_speed_norm = gnss_speed[ gnss_i ].item( ( 0, 0 ) )
			# Measurement matrix
			Z = np.matrix([
				# X position
				[ gnss_dist_x ],
				# Y position
				[ gnss_dist_y ],
				# Speed norm
				[ gnss_speed_norm ]
			])
			
			# Calculate gain
			K = P * H.transpose() * np.linalg.inv( ( H * P * H.transpose() + R ) )
			# Estimate state mean
			X = X + K * ( Z - exec_h_func( X, imu_dt ) )
			# Estimate state variance
			P = P - K * H * P
			
			gnss_i = gnss_i + 1
			
		# Control vector matrix
		U = np.matrix([
			# X accel
			[ accel.item( ( 0, 0 ) ) ],
			# Y accel
			[ accel.item( ( 1, 0 ) ) ],
			# Alpha gyro
			[ gyro.item( ( 0, 0 ) ) ]
		]) 
		# ----- Kalman predict step
		F = get_F_matrix( X, U, imu_dt )
		X = exec_f_func( X, U, imu_dt )
		P = F * P * F.transpose() + Q
		
		state_list.append( X.copy() )
		var_list.append( P.copy() )
	
	return [ state_list, var_list ]
		
	