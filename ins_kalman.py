import numpy as np
import numpy.ma as ma


def kfilter(imu_time, imu_dist, gnss_time, gnss_dist, std_acc_bias, std_gnss):
	# Output data
	dist_err_list = []
	speed_err_list = []
	bias_list = []
	
	# IMU sampling period
	imu_dt = imu_time[1] - imu_time[0]
	# State matrix
	X = np.matrix([ 
		# Accel bias
		[0.0],
		# Speed error
		[0.0],
		# Distance error
		[0.0]
	])
	# Measurement matrix
	H = np.matrix([0, 0, 1])
	# State prediction matrix
	F = np.matrix([[1, 0, 0 ],
				  [imu_dt, 1, 0],
				  [0, imu_dt, 1]])
	# Process noise matrix
	Q = np.matrix([
		[0, 0, 0],
		[0, 0, 0],
		[0, 0, 0]
	])
	# Measurement noise matrix
	R = np.matrix([std_gnss**2])
	# State covariance matrix
	P = np.matrix([[std_acc_bias**2, 0, 0],
				  [0, 0, 0],
				  [0, 0, 0]])
	
	gnss_i = 0
	for t, imu_dist_val in zip(imu_time, imu_dist):
		# Gnss data available
		if (gnss_i < len(gnss_time) and t > gnss_time[gnss_i]):			
			# ----- Kalman update step
			Z = np.matrix([imu_dist_val - gnss_dist[gnss_i]])
			# Calculate gain
			K = P * H.transpose() * np.linalg.inv( ( H * P * H.transpose() + R ) )
			# Estimate state
			X = X + K * ( Z - H * X )
			# Estimate noise
			P = P - K * H * P
			
			gnss_i = gnss_i + 1
			
		# ----- Kalman predict step
		X = F * X
		P = F * P * F.transpose() + Q
		
		dist_err_list.append(X[2, 0])
		speed_err_list.append(X[1, 0])
		bias_list.append(X[0, 0])
	
	return [bias_list, speed_err_list, dist_err_list]
		
	