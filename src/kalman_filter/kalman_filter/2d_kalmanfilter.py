import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv

class MyKalmanFilter(Node):
    def __init__(self):
        super().__init__('my_kalman_filter_node')
        # Initialize custom Kalman variables
        self.state_estimate = np.zeros((3, 1))  # Initialize state estimate (x, y, theta)
        self.state_covariance = np.eye(3)  # Initialize state covariance matrix
        
        # Custom process noise covariance matrix Q
        self.process_noise_covariance = np.diag([0.01, 0.01, 0.01])
        
        # Custom measurement noise covariance matrix R
        self.measurement_noise_covariance = np.diag([0.1, 0.1, 0.01])

        # Subscribe to the /custom_odom_noise topic
        self.subscription = self.create_subscription(Odometry,
                                                     '/odom_noise',
                                                     self.odom_callback,
                                                     1)
        
        # Publish the custom estimated reading
        self.custom_estimated_pub = self.create_publisher(Odometry,
                                                   "/odom_estimated", 1)

    def odom_callback(self, msg):
        # Extract custom position measurements from the Odometry message
        measurement = np.array([[msg.pose.pose.position.x],
                      [msg.pose.pose.position.y],
                      [msg.twist.twist.angular.z]])
    
        # Prediction step
        # Predict state estimate
        predicted_state = self.state_estimate
        # Predict state covariance
        predicted_covariance = self.state_covariance + self.process_noise_covariance
        
        # Update step
        # Calculate custom Kalman Gain
        kalman_gain = predicted_covariance @ inv(predicted_covariance + self.measurement_noise_covariance)
        # Update state estimate
        self.state_estimate = predicted_state + kalman_gain @ (measurement - predicted_state)
        # Update state covariance
        self.state_covariance = (np.eye(3) - kalman_gain) @ predicted_covariance
        
        # Publish the custom estimated reading
        self.publish_custom_estimated()
    
    def publish_custom_estimated(self):
        # Create Odometry message for custom estimated reading
        custom_estimated_msg = Odometry()
        
        custom_estimated_msg.header.stamp = self.get_clock().now().to_msg()
        custom_estimated_msg.pose.pose.position.x = self.state_estimate[0, 0]
        custom_estimated_msg.pose.pose.position.y = self.state_estimate[1, 0]
        custom_estimated_msg.twist.twist.angular.z = self.state_estimate[2, 0]
        
        # Publish the custom estimated reading
        self.custom_estimated_pub.publish(custom_estimated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MyKalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
