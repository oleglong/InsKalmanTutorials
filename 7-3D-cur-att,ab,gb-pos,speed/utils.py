import numpy as np
import math

def get_dcm( attitude_euler ):		
	cos_psi   = math.cos( attitude_euler.item( ( 0, 0 ) ) )
	sin_psi   = math.sin( attitude_euler.item( ( 0, 0 ) ) )
	cos_theta = math.cos( attitude_euler.item( ( 1, 0 ) ) )
	sin_theta = math.sin( attitude_euler.item( ( 1, 0 ) ) )
	cos_gamma = math.cos( attitude_euler.item( ( 2, 0 ) ) )
	sin_gamma = math.sin( attitude_euler.item( ( 2, 0 ) ) )
		
	C = np.matrix([
		[ cos_theta * cos_psi,    -cos_gamma * cos_psi * sin_theta + sin_gamma * sin_psi,    sin_gamma * cos_psi * sin_theta + cos_gamma * sin_psi  ],
		[ sin_theta,			  cos_gamma * cos_theta,									 -sin_gamma * cos_theta								    ],
		[ -cos_theta * sin_psi,	  cos_gamma * sin_psi * sin_theta + sin_gamma * cos_psi,	 -sin_gamma * sin_psi * sin_theta + cos_gamma * cos_psi ]
	])
	
	return C
	
def get_inv_dcm( attitude_euler ):
	return get_dcm( attitude_euler ).transpose()
	
def get_euler( atitude_dcm ):
	C11 = atitude_dcm.item( ( 0, 0 ) )
	C21 = atitude_dcm.item( ( 1, 0 ) )
	C31 = atitude_dcm.item( ( 2, 0 ) )
	C22 = atitude_dcm.item( ( 1, 1 ) )
	C23 = atitude_dcm.item( ( 1, 2 ) )
	
	if C21 > 1:
		C21 = 1
	elif C21 < -1:
		C21 = -1
		
	return np.matrix([
		# Psi
		[ math.atan2( -C31, C11 ) ],
		# Theta
		[ math.asin( C21 ) ],
		# Gamma
		[ math.atan2( -C23, C22 ) ]
	])
	
def attitude_euler_update( att_euler_prev, rot_speed, period ):
	cos_theta = np.cos( att_euler_prev.item( ( 1, 0 ) ) )
	sin_theta = np.sin( att_euler_prev.item( ( 1, 0 ) ) )
	cos_gamma = np.cos( att_euler_prev.item( ( 2, 0 ) ) )
	sin_gamma = np.sin( att_euler_prev.item( ( 2, 0 ) ) )
		
	wx = rot_speed.item( ( 0, 0 ) )
	wy = rot_speed.item( ( 1, 0 ) )
	wz = rot_speed.item( ( 2, 0 ) )
		
	tmp = ( 1 / cos_theta ) * ( wy * cos_gamma - wz * sin_gamma )
	att_new = np.matrix([
		# Psi
		[ att_euler_prev.item( ( 0, 0 ) ) + tmp * period ],
		# Theta
		[ att_euler_prev.item( ( 1, 0 ) ) + ( wy * sin_gamma + wz * cos_gamma ) * period ],
		# Gamma
		[ att_euler_prev.item( ( 2, 0 ) ) + ( wx - sin_theta * tmp ) * period ]
	])
	
	return att_new
		
def attitude_dcm_update( att_dcm_prev, rot_speed, period ):
	C11 = att_dcm_prev.item( ( 0, 0 ) )
	C12 = att_dcm_prev.item( ( 0, 1 ) )
	C13 = att_dcm_prev.item( ( 0, 2 ) )
	C21 = att_dcm_prev.item( ( 1, 0 ) )
	C22 = att_dcm_prev.item( ( 1, 1 ) )
	C23 = att_dcm_prev.item( ( 1, 2 ) )
	C31 = att_dcm_prev.item( ( 2, 0 ) )
	C32 = att_dcm_prev.item( ( 2, 1 ) )
	C33 = att_dcm_prev.item( ( 2, 2 ) )

	wx = rot_speed.item( ( 0, 0 ) )
	wy = rot_speed.item( ( 1, 0 ) )
	wz = rot_speed.item( ( 2, 0 ) )
	
	C11_new = C11 + ( C12 * wz - C13 * wy ) * period
	C12_new = C12 + ( C13 * wx - C11 * wz ) * period
	C13_new = C13 + ( C11 * wy - C12 * wx ) * period
	
	C21_new = C21 + ( C22 * wz - C23 * wy ) * period
	C22_new = C22 + ( C23 * wx - C21 * wz ) * period
	C23_new = C23 + ( C21 * wy - C22 * wx ) * period
	
	C31_new = C31 + ( C32 * wz - C33 * wy ) * period
	C32_new = C32 + ( C33 * wx - C31 * wz ) * period
	C33_new = C33 + ( C31 * wy - C32 * wx ) * period
	
	return np.matrix([
		[ C11_new, C12_new, C13_new ],
		[ C21_new, C22_new, C23_new ],
		[ C31_new, C32_new, C33_new ]
	])