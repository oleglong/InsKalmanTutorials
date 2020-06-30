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