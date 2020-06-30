import numpy as np
import utils
import math

# State prediction function
def exec_f_func( x_vect, u_vect, period ):
	# Old state data
	pos_gx   	 = x_vect.item( ( 0, 0 ) )
	pos_gy   	 = x_vect.item( ( 1, 0 ) )
	pos_gz   	 = x_vect.item( ( 2, 0 ) )
	speed_gx	 = x_vect.item( ( 3, 0 ) )
	speed_gy 	 = x_vect.item( ( 4, 0 ) )
	speed_gz 	 = x_vect.item( ( 5, 0 ) )
	accel_bias_x = x_vect.item( ( 6, 0 ) )
	accel_bias_y = x_vect.item( ( 7, 0 ) )
	accel_bias_z = x_vect.item( ( 8, 0 ) )
	w_bias_x 	 = x_vect.item( ( 9, 0 ) )
	w_bias_y 	 = x_vect.item( ( 10, 0 ) )
	w_bias_z     = x_vect.item( ( 11, 0 ) )
	C11 		 = x_vect.item( ( 12, 0 ) )
	C12 	 	 = x_vect.item( ( 13, 0 ) )
	C13		   	 = x_vect.item( ( 14, 0 ) )
	C21 		 = x_vect.item( ( 15, 0 ) )
	C22 	 	 = x_vect.item( ( 16, 0 ) )
	C23		   	 = x_vect.item( ( 17, 0 ) )
	C31 		 = x_vect.item( ( 18, 0 ) )
	C32 	 	 = x_vect.item( ( 19, 0 ) )
	C33		   	 = x_vect.item( ( 20, 0 ) )
	
	# Old attitude
	attitude_dcm = np.matrix([
		[ C11, C12, C13 ],
		[ C21, C22, C23 ],
		[ C31, C32, C33 ]
	])
	
	# Calibrated IMU data
	e_acc_ix = u_vect.item( ( 0, 0 ) ) - accel_bias_x
	e_acc_iy = u_vect.item( ( 1, 0 ) ) - accel_bias_y
	e_acc_iz = u_vect.item( ( 2, 0 ) ) - accel_bias_z
	e_wx = u_vect.item( ( 3, 0 ) ) - w_bias_x
	e_wy = u_vect.item( ( 4, 0 ) ) - w_bias_y
	e_wz = u_vect.item( ( 5, 0 ) ) - w_bias_z
	
	# Global accel
	accel_g = attitude_dcm * np.matrix([
		[ e_acc_ix ],
		[ e_acc_iy ],
		[ e_acc_iz ]
	])	
	accel_gx = accel_g.item( ( 0, 0 ) )
	accel_gy = accel_g.item( ( 1, 0 ) )
	accel_gz = accel_g.item( ( 2, 0 ) )
	
	# New attitude
	attitude_new = utils.attitude_dcm_update(
		attitude_dcm,
		np.matrix([
			[ e_wx ],
			[ e_wy ],
			[ e_wz ]
		]),
		period
	)	
			
	dt2 = 0.5 * period**2	
	dt = period
	
	# New state data
	return np.matrix([
		[ pos_gx + speed_gx * dt + accel_gx * dt2 ],
		[ pos_gy + speed_gy * dt + accel_gy * dt2 ],
		[ pos_gz + speed_gz * dt + accel_gz * dt2 ],
		[ speed_gx + accel_gx * dt ],
		[ speed_gy + accel_gy * dt ],
		[ speed_gz + accel_gz * dt ],
		[ accel_bias_x ],
		[ accel_bias_y ],
		[ accel_bias_z ],
		[ w_bias_x ],
		[ w_bias_y ],
		[ w_bias_z ],
		[ attitude_new.item( ( 0, 0 ) ) ],
		[ attitude_new.item( ( 0, 1 ) ) ],
		[ attitude_new.item( ( 0, 2 ) ) ],
		[ attitude_new.item( ( 1, 0 ) ) ],
		[ attitude_new.item( ( 1, 1 ) ) ],
		[ attitude_new.item( ( 1, 2 ) ) ],
		[ attitude_new.item( ( 2, 0 ) ) ],
		[ attitude_new.item( ( 2, 1 ) ) ],
		[ attitude_new.item( ( 2, 2 ) ) ]
	])
	
# State prediction Jacobian matrix
def get_F_matrix( x_vect, u_vect, period ):
	# Old state data
	pos_gx   	 = x_vect.item( ( 0, 0 ) )
	pos_gy   	 = x_vect.item( ( 1, 0 ) )
	pos_gz   	 = x_vect.item( ( 2, 0 ) )
	speed_gx	 = x_vect.item( ( 3, 0 ) )
	speed_gy 	 = x_vect.item( ( 4, 0 ) )
	speed_gz 	 = x_vect.item( ( 5, 0 ) )
	accel_bias_x = x_vect.item( ( 6, 0 ) )
	accel_bias_y = x_vect.item( ( 7, 0 ) )
	accel_bias_z = x_vect.item( ( 8, 0 ) )
	w_bias_x 	 = x_vect.item( ( 9, 0 ) )
	w_bias_y 	 = x_vect.item( ( 10, 0 ) )
	w_bias_z     = x_vect.item( ( 11, 0 ) )
	C11 		 = x_vect.item( ( 12, 0 ) )
	C12 	 	 = x_vect.item( ( 13, 0 ) )
	C13		   	 = x_vect.item( ( 14, 0 ) )
	C21 		 = x_vect.item( ( 15, 0 ) )
	C22 	 	 = x_vect.item( ( 16, 0 ) )
	C23		   	 = x_vect.item( ( 17, 0 ) )
	C31 		 = x_vect.item( ( 18, 0 ) )
	C32 	 	 = x_vect.item( ( 19, 0 ) )
	C33		   	 = x_vect.item( ( 20, 0 ) )
	
	# Calibrated IMU data
	e_acc_ix = u_vect.item( ( 0, 0 ) ) - accel_bias_x
	e_acc_iy = u_vect.item( ( 1, 0 ) ) - accel_bias_y
	e_acc_iz = u_vect.item( ( 2, 0 ) ) - accel_bias_z
	e_wx = u_vect.item( ( 3, 0 ) ) - w_bias_x
	e_wy = u_vect.item( ( 4, 0 ) ) - w_bias_y
	e_wz = u_vect.item( ( 5, 0 ) ) - w_bias_z
	
	# accel_gx = C11 * e_acc_ix + C12 * e_acc_iy + C13 * e_acc_iz
	# accel_gy = C21 * e_acc_ix + C22 * e_acc_iy + C23 * e_acc_iz
	# accel_gz = C31 * e_acc_ix + C32 * e_acc_iy + C33 * e_acc_iz	
	
	dt2 = 0.5 * period**2	
	dt = period
	
	F = np.matrix([
	#    rx   ry   rz    vx   vy    vz   abx          aby          abz         wbx            wby               wbz             C11              C12              C13              C21              C22              C23             C31              C32              C33             
		[1,   0,   0,    dt,  0,   0,   -C11 * dt2,  -C12 * dt2,  -C13 * dt2,  0,              0,               0,              e_acc_ix * dt2,  e_acc_iy * dt2,  e_acc_iz * dt2,  0,               0,               0,              0,               0,               0,             ],
		[0,   1,   0,    0,   dt,  0,   -C21 * dt2,  -C22 * dt2,  -C23 * dt2,  0,              0,               0,              0,  			    0,			     0,			   e_acc_ix * dt2,  e_acc_iy * dt2,  e_acc_iz * dt2, 0,  			  0,		       0,             ],
		[0,   0,   1,    0,   0,   dt,  -C31 * dt2,  -C32 * dt2,  -C33 * dt2,  0,              0,               0,              0,  			    0,			     0,            0,  		        0,				 0,              e_acc_ix * dt2,  e_acc_iy * dt2,  e_acc_iz * dt2 ],
		                                                                                                                        
		[0,   0,   0,    1,   0,   0,   -C11 * dt,   -C12 * dt,   -C13 * dt,   0,              0,               0,              e_acc_ix * dt,   e_acc_iy * dt,   e_acc_iz * dt ,  0,               0,               0,              0,               0,               0,             ], 
		[0,   0,   0,    0,   1,   0,   -C21 * dt,   -C22 * dt,   -C23 * dt,   0,              0,               0,              0,  			    0,			     0,			   e_acc_ix * dt ,  e_acc_iy * dt ,  e_acc_iz * dt,  0,  			  0,			   0,             ], 
		[0,   0,   0,    0,   0,   1,   -C31 * dt,   -C32 * dt,   -C33 * dt,   0,              0,               0,              0,  			    0,			     0, 		   0,  			    0,				 0,              e_acc_ix * dt ,  e_acc_iy * dt ,  e_acc_iz * dt  ], 
		                                                                                                                        
		[0,   0,   0,    0,   0,   0,   1,           0,           0,           0,              0,               0,              0,               0,               0,               0,               0,               0,              0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           1,           0,           0,              0,               0,              0,               0,               0,               0,               0,               0,              0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           0,           1,           0,              0,               0,              0,               0,               0,               0,               0,               0,              0,               0,               0              ],
		                                                                                                                        
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           1,              0,               0,              0,               0,               0,               0,               0,               0,              0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           0,              1,               0,              0,               0,               0,               0,               0,               0,              0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           0,              0,               1,              0,               0,               0,               0,               0,               0,              0,               0,               0              ],
		                                                                                                                        
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           0,              C13 * period,    -C12 * period,  1,               e_wz * dt,       -e_wy * dt,      0,               0,               0,              0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           -C13 * period,  0,               C11 * period,   -e_wz * dt,      1,               e_wx * dt,       0,               0,               0,              0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           C12 * period,   -C11 * period,   0,              e_wy * dt,       -e_wx * dt,      1,               0,               0,               0,              0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           0,              C23 * period,    -C22 * period,  0,               0,               0,               1,               e_wz * dt,       -e_wy * dt,     0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           -C23 * period,  0,               C21 * period,   0,               0,               0,               -e_wz * dt,      1,               e_wx * dt,      0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           C22 * period,   -C21 * period,   0,              0,               0,               0,               e_wy * dt,       -e_wx * dt,      1,              0,               0,               0              ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           0,              C33 * period,    -C32 * period,  0,               0,               0,               0,               0,               0,              1,               e_wz * dt,       -e_wy * dt,    ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           -C33 * period,  0,               C31 * period,   0,               0,               0,               0,               0,               0,              -e_wz * dt,      1,               e_wx * dt,     ],
		[0,   0,   0,    0,   0,   0,   0,           0,           0,           C32 * period,   -C31 * period,   0,              0,               0,               0,               0,               0,               0,              e_wy * dt,       -e_wx * dt,      1,             ],
	])
	
	return F
	
# Measurement function
def exec_h_func( x_vect, period ):
	pos_gx   	 = x_vect.item( ( 0, 0 ) )
	pos_gy   	 = x_vect.item( ( 1, 0 ) )
	pos_gz   	 = x_vect.item( ( 2, 0 ) )
	speed_gx	 = x_vect.item( ( 3, 0 ) )
	speed_gy 	 = x_vect.item( ( 4, 0 ) )
	speed_gz 	 = x_vect.item( ( 5, 0 ) )
	
	speed_norm   = np.sqrt( speed_gx**2 + speed_gy**2 + speed_gz**2 )
	
	return np.matrix([
		[ pos_gx ],
		[ pos_gy ],
		[ pos_gz ],
		[ speed_norm ]
	])	
	
# Measurement Jacobian matrix
def get_H_matrix( x_vect, period ):
	speed_gx	 = x_vect.item( ( 3, 0 ) )
	speed_gy 	 = x_vect.item( ( 4, 0 ) )
	speed_gz 	 = x_vect.item( ( 5, 0 ) )
	speed_norm   = np.sqrt( speed_gx**2 + speed_gy**2 + speed_gz**2 )
	
	# d(speed_norm)/d(speed_gx)
	d_sn_d_sgx = speed_gx / speed_norm
	# d(speed_norm)/d(speed_gy)
	d_sn_d_sgy = speed_gy / speed_norm
	# d(speed_norm)/d(speed_gz)
	d_sn_d_sgz = speed_gz / speed_norm

	return np.matrix([
		[ 1,  0,  0,  0,          0,           0,           0,  0,  0,   0,  0,  0,   0,  0,  0,   0,  0,  0,   0,  0,  0 ],
		[ 0,  1,  0,  0,          0,           0,           0,  0,  0,   0,  0,  0,   0,  0,  0,   0,  0,  0,   0,  0,  0 ],
		[ 0,  0,  1,  0,          0,           0,           0,  0,  0,   0,  0,  0,   0,  0,  0,   0,  0,  0,   0,  0,  0 ],
		[ 0,  0,  0,  d_sn_d_sgx, d_sn_d_sgy,  d_sn_d_sgz,  0,  0,  0,   0,  0,  0,   0,  0,  0,   0,  0,  0,   0,  0,  0 ]
	])
	
def ins_ext_kfilter( imu_time, imu_accel, imu_gyro, accel_bias_std, gyro_bias_std,
					 attitude0, attitude0_std, 
					 gnss_time, gnss_speed, gnss_dist, gnss_speed_std, gnss_dist_std ):
	# Output data
	state_list = []
	var_list = []
	
	# IMU sampling period
	imu_dt = imu_time[1] - imu_time[0]
	
	dcm0 = utils.get_dcm( attitude0 )
	# State matrix
	X = np.matrix([ 
		# X position
		[0.0],
		# Y position
		[0.0],
		# Z position
		[0.0],
		# X speed
		[0.0],
		# Y speed
		[0.0],
		# Z speed
		[0.0],
		# Accel X bias
		[0.0],
		# Accel Y bias
		[0.0],
		# Accel Z bias
		[0.0],
		# Gyro X bias
		[0.0],
		# Gyro Y bias
		[0.0],
		# Gyro Z bias
		[0.0],
		# C11
		[ dcm0.item( ( 0, 0 ) ) ],
		# C12
		[ dcm0.item( ( 0, 1 ) ) ],
		# C13
		[ dcm0.item( ( 0, 2 ) ) ],
		# C21
		[ dcm0.item( ( 1, 0 ) ) ],
		# C22
		[ dcm0.item( ( 1, 1 ) ) ],
		# C23
		[ dcm0.item( ( 1, 2 ) ) ],
		# C31
		[ dcm0.item( ( 2, 0 ) ) ],
		# C32
		[ dcm0.item( ( 2, 1 ) ) ],
		# C33
		[ dcm0.item( ( 2, 2 ) ) ]
	])
	# Process noise matrix
	Q = np.matrix([
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],		
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],		
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],		
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],		
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	])
	# Measurement noise matrix
	R = np.matrix([
		[gnss_dist_std**2, 0, 0,  0 ],
		[0, gnss_dist_std**2, 0,  0 ],
		[0, 0, gnss_dist_std**2,  0 ],
		[0, 0, 0, gnss_speed_std**2 ]
	])	
	'''
	dcm_noise_0 = np.square( 
		utils.get_dcm( np.matrix([
			# Psi
			[ attitude0_std ],
			# Theta
			[ attitude0_std ],
			# Gamma
			[ attitude0_std ]
		]) )
	)
	'''
	'''
	dcm_noise_0 = np.matrix([
		[ 1.0, 1.0, 1.0 ],
		[ 1.0, 1.0, 1.0 ],
		[ 1.0, 1.0, 1.0 ]
	])
	'''
	# State covariance matrix
	'''
	P = np.matrix([
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		                                                                
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		                                                                
		[0, 0, 0, 0, 0, 0, accel_bias_std**2, 0, 0, 0, 0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, accel_bias_std**2, 0, 0, 0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, accel_bias_std**2, 0, 0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		                                                                
		[0, 0, 0, 0, 0, 0, 0, 0, 0, gyro_bias_std**2,  0, 0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gyro_bias_std**2,  0,            0, 0, 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gyro_bias_std**2,             0, 0, 0,  0, 0, 0,  0, 0, 0],
		
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dcm_noise_0.item( ( 0, 0 ) ), 0, 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dcm_noise_0.item( ( 0, 1 ) ), 0,  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dcm_noise_0.item( ( 0, 2 ) ),  0, 0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dcm_noise_0.item( ( 1, 0 ) ),  0, 0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dcm_noise_0.item( ( 1, 1 ) ),  0,  0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dcm_noise_0.item( ( 1, 2 ) ),   0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dcm_noise_0.item( ( 2, 0 ) ),   0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dcm_noise_0.item( ( 2, 1 ) ),   0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dcm_noise_0.item( ( 2, 2 ) ) ],
	])
	'''
	dcm_c = 0.05
	P = np.matrix([
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		                                                                        
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		                                                                        
		[0, 0, 0, 0, 0, 0, accel_bias_std**2, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, accel_bias_std**2, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, accel_bias_std**2, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		                                                                        
		[0, 0, 0, 0, 0, 0, 0, 0, 0, gyro_bias_std**2,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gyro_bias_std**2,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gyro_bias_std**2,   0, 0, 0, 0, 0, 0, 0, 0, 0],
		
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c ],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c ],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c ],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c ],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c ],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c ],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c ],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c ],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c,  dcm_c, dcm_c, dcm_c ],
	])
		
	gnss_i = 0
	for t, accel, gyro in zip( imu_time, imu_accel, imu_gyro ):			
		# ----- Kalman predict step		
		# Control vector matrix
		U = np.matrix([
			# X accel
			[ accel.item( ( 0, 0 ) ) ],
			# Y accel
			[ accel.item( ( 1, 0 ) ) ],
			# Z accel
			[ accel.item( ( 2, 0 ) ) ],
			# X gyro
			[ gyro.item( ( 0, 0 ) ) ],
			# Y gyro
			[ gyro.item( ( 1, 0 ) ) ],
			# Z gyro
			[ gyro.item( ( 2, 0 ) ) ]
		]) 
		F = get_F_matrix( X, U, imu_dt )
		X = exec_f_func( X, U, imu_dt )
		P = F * P * F.transpose() + Q
		
		# Gnss data is available
		if ( gnss_i < len( gnss_time ) and t > gnss_time[ gnss_i ] ):			
			# ----- Kalman update step
			# Measurement matrix
			Z = np.matrix([
				# X position
				[ gnss_dist[ gnss_i ].item( ( 0, 0 ) ) ],
				# Y position
				[ gnss_dist[ gnss_i ].item( ( 1, 0 ) ) ],
				# Z position
				[ gnss_dist[ gnss_i ].item( ( 2, 0 ) ) ],
				# Speed norm
				[ gnss_speed[ gnss_i ].item( ( 0, 0 ) ) ]
			])
			H = get_H_matrix( X, imu_dt )		
			# Calculate gain
			K = P * H.transpose() * np.linalg.inv( ( H * P * H.transpose() + R ) )
			# Estimate state mean
			X = X + K * ( Z - exec_h_func( X, imu_dt ) )
			# Estimate state variance
			P = P - K * H * P
			
			gnss_i = gnss_i + 1
			
		state_list.append( X.copy() )
		var_list.append( P.copy() )
	
	return [ state_list, var_list ]
		
	