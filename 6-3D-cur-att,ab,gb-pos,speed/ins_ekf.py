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
	psi 		 = x_vect.item( ( 12, 0 ) )
	theta 	 	 = x_vect.item( ( 13, 0 ) )
	gamma   	 = x_vect.item( ( 14, 0 ) )
	
	# Calibrated IMU data
	est_accel_ix = u_vect.item( ( 0, 0 ) ) - accel_bias_x
	est_accel_iy = u_vect.item( ( 1, 0 ) ) - accel_bias_y
	est_accel_iz = u_vect.item( ( 2, 0 ) ) - accel_bias_z
	est_wx = u_vect.item( ( 3, 0 ) ) - w_bias_x
	est_wy = u_vect.item( ( 4, 0 ) ) - w_bias_y
	est_wz = u_vect.item( ( 5, 0 ) ) - w_bias_z
	
	# Global accel
	accel_g = utils.get_dcm( np.matrix([
		[ psi ], 
		[ theta ],
		[ gamma ]
	])) * np.matrix([
		[ est_accel_ix ],
		[ est_accel_iy ],
		[ est_accel_iz ]
	])	
	# Substract gravity component
	accel_g = accel_g - np.matrix([
		# X
		[ 0 ],
		# Y
		[ 9.81 ],
		# Z
		[ 0 ]
	])
	accel_gx = accel_g.item( ( 0, 0 ) )
	accel_gy = accel_g.item( ( 1, 0 ) )
	accel_gz = accel_g.item( ( 2, 0 ) )
	
	# New attitude
	attitude_new = utils.attitude_euler_update(
		np.matrix([
			[ psi ],
			[ theta ],
			[ gamma ]
		]),
		np.matrix([
			[ est_wx ],
			[ est_wy ],
			[ est_wz ]
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
		# Psi
		[ attitude_new.item( ( 0, 0 ) ) ],
		# Theta
		[ attitude_new.item( ( 1, 0 ) ) ],
		# Gamma
		[ attitude_new.item( ( 2, 0 ) ) ]
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
	psi 		 = x_vect.item( ( 12, 0 ) )
	theta 	 	 = x_vect.item( ( 13, 0 ) )
	gamma   	 = x_vect.item( ( 14, 0 ) )
	
	dcm = utils.get_dcm( np.matrix([
		[ psi ], 
		[ theta ],
		[ gamma ]
	]))
	
	cos_psi   = math.cos( psi )
	sin_psi   = math.sin( psi )
	cos_theta = math.cos( theta )
	sin_theta = math.sin( theta )
	cos_gamma = math.cos( gamma )
	sin_gamma = math.sin( gamma )
	
	# Calibrated IMU data
	est_accel_ix = u_vect.item( ( 0, 0 ) ) - accel_bias_x
	est_accel_iy = u_vect.item( ( 1, 0 ) ) - accel_bias_y
	est_accel_iz = u_vect.item( ( 2, 0 ) ) - accel_bias_z
	est_wx = u_vect.item( ( 3, 0 ) ) - w_bias_x
	est_wy = u_vect.item( ( 4, 0 ) ) - w_bias_y
	est_wz = u_vect.item( ( 5, 0 ) ) - w_bias_z
	
	# d(C11)/d(psi)
	d_c11_d_psi = cos_theta * ( -sin_psi )
	# d(C12)/d(psi)
	d_c12_d_psi = -cos_gamma * ( -sin_psi ) * sin_theta + sin_gamma * cos_psi
	# d(C13)/d(psi)
	d_c13_d_psi = sin_gamma * ( -sin_psi ) * sin_theta + cos_gamma * cos_psi
	# d(C21)/d(psi)
	d_c21_d_psi = 0
	# d(C22)/d(psi)
	d_c22_d_psi = 0
	# d(C23)/d(psi)
	d_c23_d_psi = 0
	# d(C31)/d(psi)
	d_c31_d_psi = -cos_theta * cos_psi
	# d(C32)/d(psi)
	d_c32_d_psi = cos_gamma * cos_psi * sin_theta + sin_gamma * ( -sin_psi )
	# d(C33)/d(psi)
	d_c33_d_psi = -sin_gamma * cos_psi * sin_theta + cos_gamma * ( -sin_psi )
	
	# d(C11)/d(theta)
	d_c11_d_theta = ( -sin_theta ) * cos_psi
	# d(C12)/d(theta)
	d_c12_d_theta = -cos_gamma * cos_psi * cos_theta
	# d(C13)/d(theta)
	d_c13_d_theta = sin_gamma * cos_psi * cos_theta
	# d(C21)/d(theta)
	d_c21_d_theta = cos_theta
	# d(C22)/d(theta)
	d_c22_d_theta = cos_gamma * ( -sin_theta )
	# d(C23)/d(theta)
	d_c23_d_theta = -sin_gamma * ( -sin_theta )
	# d(C31)/d(theta)
	d_c31_d_theta = -( -sin_theta ) * sin_psi
	# d(C32)/d(theta)
	d_c32_d_theta = cos_gamma * sin_psi * cos_theta
	# d(C33)/d(theta)
	d_c33_d_theta = -sin_gamma * sin_psi * cos_theta
	
	# d(C11)/d(gamma)
	d_c11_d_gamma = 0
	# d(C12)/d(gamma)
	d_c12_d_gamma = -( -sin_gamma ) * cos_psi * sin_theta + cos_gamma * sin_psi
	# d(C13)/d(gamma)
	d_c13_d_gamma = cos_gamma * cos_psi * sin_theta + ( -sin_gamma ) * sin_psi
	# d(C21)/d(gamma)
	d_c21_d_gamma = 0
	# d(C22)/d(gamma)
	d_c22_d_gamma = ( -sin_gamma ) * cos_theta
	# d(C23)/d(gamma)
	d_c23_d_gamma = -cos_gamma * cos_theta
	# d(C31)/d(gamma)
	d_c31_d_gamma = 0
	# d(C32)/d(gamma)
	d_c32_d_gamma = ( -sin_gamma ) * sin_psi * sin_theta + cos_gamma * cos_psi
	# d(C33)/d(gamma)
	d_c33_d_gamma = -cos_gamma * sin_psi * sin_theta + ( -sin_gamma ) * cos_psi
	
	# accel_gx = C11 * est_accel_ix + C12 * est_accel_iy + C13 * est_accel_iz
	# accel_gy = C21 * est_accel_ix + C22 * est_accel_iy + C23 * est_accel_iz
	# accel_gz = C31 * est_accel_ix + C32 * est_accel_iy + C33 * est_accel_iz
	
	# d(accel_gx)/d(accel_bias_x) = -C11
	d_agx_d_abx = -dcm.item( ( 0, 0 ) )
	# d(accel_gx)/d(accel_bias_y) = -C12
	d_agx_d_aby = -dcm.item( ( 0, 1 ) )
	# d(accel_gx)/d(accel_bias_z) = -C13
	d_agx_d_abz = -dcm.item( ( 0, 2 ) )
	# d(accel_gx)/d(psi)
	d_agx_d_psi   = d_c11_d_psi   * est_accel_ix  +  d_c12_d_psi   * est_accel_iy  +  d_c13_d_psi   * est_accel_iz
	# d(accel_gx)/d(theta)
	d_agx_d_theta = d_c11_d_theta * est_accel_ix  +  d_c12_d_theta * est_accel_iy  +  d_c13_d_theta * est_accel_iz
	# d(accel_gx)/d(gamma)
	d_agx_d_gamma = d_c11_d_gamma * est_accel_ix  +  d_c12_d_gamma * est_accel_iy  +  d_c13_d_gamma * est_accel_iz
	
	# d(accel_gy)/d(accel_bias_x) = -C21
	d_agy_d_abx = -dcm.item( ( 1, 0 ) )
	# d(accel_gy)/d(accel_bias_y) = -C22
	d_agy_d_aby = -dcm.item( ( 1, 1 ) )
	# d(accel_gy)/d(accel_bias_z) = -C23
	d_agy_d_abz = -dcm.item( ( 1, 2 ) )
	# d(accel_gy)/d(psi)
	d_agy_d_psi   = d_c21_d_psi   * est_accel_ix  +  d_c22_d_psi   * est_accel_iy  +  d_c23_d_psi   * est_accel_iz
	# d(accel_gy)/d(theta)
	d_agy_d_theta = d_c21_d_theta * est_accel_ix  +  d_c22_d_theta * est_accel_iy  +  d_c23_d_theta * est_accel_iz
	# d(accel_gy)/d(gamma)
	d_agy_d_gamma = d_c21_d_gamma * est_accel_ix  +  d_c22_d_gamma * est_accel_iy  +  d_c23_d_gamma * est_accel_iz
		
	# d(accel_gz)/d(accel_bias_x) = -C31
	d_agz_d_abx = -dcm.item( ( 2, 0 ) )
	# d(accel_gz)/d(accel_bias_y) = -C32
	d_agz_d_aby = -dcm.item( ( 2, 1 ) )
	# d(accel_gz)/d(accel_bias_z) = -C33
	d_agz_d_abz = -dcm.item( ( 2, 2 ) )
	# d(accel_gz)/d(psi)
	d_agz_d_psi   = d_c31_d_psi   * est_accel_ix  +  d_c32_d_psi   * est_accel_iy  +  d_c33_d_psi   * est_accel_iz
	# d(accel_gz)/d(theta)
	d_agz_d_theta = d_c31_d_theta * est_accel_ix  +  d_c32_d_theta * est_accel_iy  +  d_c33_d_theta * est_accel_iz
	# d(accel_gz)/d(gamma)
	d_agz_d_gamma = d_c31_d_gamma * est_accel_ix  +  d_c32_d_gamma * est_accel_iy  +  d_c33_d_gamma * est_accel_iz
	
	# psi[i+1]   = psi[i] + ( 1 / cos(theta) ) * ( wy * cos(gamma) - wz * sin(gamma) ) * period
	# theta[i+1] = theta[i] + ( wy * sin(gamma) + wz * cos(gamma) ) * period
	# gamma[i+1] = gamma[i] + ( wx - sin(theta) / cos(theta) * ( wy * cos(gamma) - wz * sin(gamma) ) ) * period
	# d(wx)/d(wbx) = -1
	# d(wy)/d(wby) = -1
	# d(wz)/d(wbz) = -1
	d_psi_d_wbx     = 0
	d_psi_d_wby     = 1 / cos_theta * ( -cos_gamma ) * period
	d_psi_d_wbz     = 1 / cos_theta * ( -( -sin_gamma ) ) * period
	d_psi_d_psi	    = 1
	d_psi_d_theta   = sin_theta / ( cos_theta ** 2 ) * ( est_wy * cos_gamma - est_wz * sin_gamma ) * period
	d_psi_d_gamma   = 1 / cos_theta * ( est_wy * ( -sin_gamma ) - est_wz * cos_gamma ) * period
	
	d_theta_d_wbx   = 0
	d_theta_d_wby   = -sin_gamma * period
	d_theta_d_wbz   = -cos_gamma * period
	d_theta_d_psi	= 0
	d_theta_d_theta = 1
	d_theta_d_gamma = ( est_wy * cos_gamma + est_wz * ( -sin_gamma ) ) * period
	
	d_gamma_d_wbx   = -period
	d_gamma_d_wby   = -sin_theta / cos_theta * ( -cos_gamma ) * period
	d_gamma_d_wbz   = -sin_theta / cos_theta * ( -( -sin_gamma ) ) * period
	d_gamma_d_psi	= 0
	d_gamma_d_theta = -1 / ( cos_theta ** 2 ) * ( est_wy * cos_gamma - est_wz * sin_gamma ) * period
	d_gamma_d_gamma = 1 - sin_theta / cos_theta * ( est_wy * ( -sin_gamma ) - est_wz * cos_gamma ) * period
	
	dt2 = 0.5 * period**2	
	dt = period
	
	F = np.matrix([
	#    rx   ry   rz    vx   vy    vz   abx                 aby                 abz                 wbx             wby             wbz             psi                 theta                 gamma
		[1,   0,   0,    dt,   0,   0,   d_agx_d_abx * dt2,  d_agx_d_aby * dt2,  d_agx_d_abz * dt2,  0,              0,              0,              d_agx_d_psi * dt2,  d_agx_d_theta * dt2,  d_agx_d_gamma * dt2 ],
		[0,   1,   0,    0,    dt,  0,   d_agy_d_abx * dt2,  d_agy_d_aby * dt2,  d_agy_d_abz * dt2,  0,              0,              0,              d_agy_d_psi * dt2,  d_agy_d_theta * dt2,  d_agy_d_gamma * dt2 ],
		[0,   0,   1,    0,    0,   dt,  d_agz_d_abx * dt2,  d_agz_d_aby * dt2,  d_agz_d_abz * dt2,  0,              0,              0,              d_agz_d_psi * dt2,  d_agz_d_theta * dt2,  d_agz_d_gamma * dt2 ],
		[0,   0,   0,    1,    0,   0,   d_agx_d_abx * dt,   d_agx_d_aby * dt,   d_agx_d_abz * dt,   0,              0,              0,              d_agx_d_psi * dt,   d_agx_d_theta * dt,   d_agx_d_gamma * dt  ],
		[0,   0,   0,    0,    1,   0,   d_agy_d_abx * dt,   d_agy_d_aby * dt,   d_agy_d_abz * dt,	 0,              0,              0,              d_agy_d_psi * dt,   d_agy_d_theta * dt,   d_agy_d_gamma * dt  ],
		[0,   0,   0,    0,    0,   1,   d_agz_d_abx * dt,   d_agz_d_aby * dt,   d_agz_d_abz * dt,	 0,              0,              0,              d_agz_d_psi * dt,   d_agz_d_theta * dt,   d_agz_d_gamma * dt  ],
		[0,   0,   0,    0,    0,   0,   1,                  0,                  0,                  0,              0,              0,              0,                  0,                    0                   ],
		[0,   0,   0,    0,    0,   0,   0,                  1,                  0,                  0,              0,              0,              0,                  0,                    0                   ],
		[0,   0,   0,    0,    0,   0,   0,                  0,                  1,                  0,              0,              0,              0,                  0,                    0                   ],
		[0,   0,   0,    0,    0,   0,   0,                  0,                  0,                  1,              0,              0,              0,                  0,                    0                   ],
		[0,   0,   0,    0,    0,   0,   0,                  0,                  0,                  0,              1,              0,              0,                  0,                    0                   ],
		[0,   0,   0,    0,    0,   0,   0,                  0,                  0,                  0,              0,              1,              0,                  0,                    0                   ],
		[0,   0,   0,    0,    0,   0,   0,                  0,                  0,                  d_psi_d_wbx,    d_psi_d_wby,    d_psi_d_wbz,    d_psi_d_psi,        d_psi_d_theta,        d_psi_d_gamma       ],
		[0,   0,   0,    0,    0,   0,   0,                  0,                  0,                  d_theta_d_wbx,  d_theta_d_wby,  d_theta_d_wbz,  d_theta_d_psi,      d_theta_d_theta,      d_theta_d_gamma     ],
		[0,   0,   0,    0,    0,   0,   0,                  0,                  0,                  d_gamma_d_wbx,  d_gamma_d_wby,  d_gamma_d_wbz,  d_gamma_d_psi,      d_gamma_d_theta,      d_gamma_d_gamma     ],
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
	speed_norm   = np.sqrt( speed_gx**2 + speed_gy**2 + speed_gz**2 ) + 0.000000001
	
	# d(speed_norm)/d(speed_gx)
	d_sn_d_sgx = speed_gx / speed_norm
	# d(speed_norm)/d(speed_gy)
	d_sn_d_sgy = speed_gy / speed_norm
	# d(speed_norm)/d(speed_gz)
	d_sn_d_sgz = speed_gz / speed_norm

	return np.matrix([
		[ 1,  0,  0,  0,          0,           0,           0,  0,  0,  0,  0,  0,  0,  0,  0 ],
		[ 0,  1,  0,  0,          0,           0,           0,  0,  0,  0,  0,  0,  0,  0,  0 ],
		[ 0,  0,  1,  0,          0,           0,           0,  0,  0,  0,  0,  0,  0,  0,  0 ],
		[ 0,  0,  0,  d_sn_d_sgx, d_sn_d_sgy,  d_sn_d_sgz,  0,  0,  0,  0,  0,  0,  0,  0,  0 ]
	])
	
def ins_ext_kfilter( imu_time, imu_accel, imu_gyro, accel_bias_std, accel_w_std, gyro_bias_std, gyro_w_std,
					 attitude0, attitude0_std, gyro_bias0, 
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
		[ gyro_bias0.item( ( 0, 0 ) ) ],
		# Gyro Y bias
		[ gyro_bias0.item( ( 1, 0 ) ) ],
		# Gyro Z bias
		[ gyro_bias0.item( ( 2, 0 ) ) ],
		# Psi
		[ attitude0.item( ( 0, 0 ) ) ],
		# Theta
		[ attitude0.item( ( 1, 0 ) ) ],
		# Gamma
		[ attitude0.item( ( 2, 0 ) ) ]
	])
	# Process noise matrix
	pos_q_std = accel_w_std * imu_dt**2 / 2
	speed_q_std = accel_w_std * imu_dt
	angle_q_std = gyro_w_std * imu_dt
	Q = np.matrix([
		[pos_q_std**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, pos_q_std**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, pos_q_std**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, speed_q_std**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, speed_q_std**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, speed_q_std**2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, angle_q_std**2, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, angle_q_std**2, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, angle_q_std**2]
	])
	'''
	Q = np.matrix([
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	])
	'''
	# Measurement noise matrix
	R = np.matrix([
		[gnss_dist_std**2, 0, 0,  0 ],
		[0, gnss_dist_std**2, 0,  0 ],
		[0, 0, gnss_dist_std**2,  0 ],
		[0, 0, 0, gnss_speed_std**2 ]
	])
	# State covariance matrix
	P = np.matrix([
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0, 0, 0, 0],
		
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0,                 0, 0, 0, 0, 0, 0, 0, 0],
		
		[0, 0, 0, 0, 0, 0, accel_bias_std**2, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, accel_bias_std**2, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, accel_bias_std**2, 0, 0, 0, 0, 0, 0],
		
		[0, 0, 0, 0, 0, 0, 0, 0, 0, gyro_w_std**2,  0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gyro_w_std**2,  0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gyro_w_std**2,  0, 0, 0],

		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, attitude0_std**2,  0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, attitude0_std**2,  0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, attitude0_std**2 ]
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
		
	