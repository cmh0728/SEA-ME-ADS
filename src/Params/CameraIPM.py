import rclpy
import tf2_ros as tf2
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PointStamped
from ipm_library.exceptions import NoIntersectionError
from ipm_library.ipm import IPM
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from shape_msgs.msg import Plane
from std_msgs.msg import Header
from vision_msgs.msg import Point2D


class IPMExample(Node):
    def __init__(self):
        # Let's initialize our node
        super().__init__('ipm_example')

        # We will need to provide the camera's intrinsic parameters to perform the projection
        # In a real scenario, this would be provided by the camera driver on a topic
        # If you don't know the intrinsic parameters of your camera,
        # you can use the camera_calibration ROS package to calibrate your camera
        self.camera_info = CameraInfo(
            header=Header(
                # This defines where the camera is located on the robot
                frame_id='camera_optical_frame',
            ),
            width=2048,
            height=1536,
            k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.],
            d=[0., 0., 0., 0., 0.] # The distortion coefficients are optional
        )

        # We want to publish the projected points on a topic so we can visualize them in RViz
        self.point_pub = self.create_publisher(PointStamped, 'ipm_point', 10)

        # A TF buffer is needed as we need to know the spatial relationship between the camera and the plane
        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self, spin_thread=True)

        # Initialize the IPM library with a reference to the forward kinematics of the robot
        # We also need to provide the camera info, this is optional during the initialization
        # as it can be provided via a setter later on as well
        # We can also set the distortion parameter to True if we want to use the distortion model
        # from the camera info and don't want to rectify the image beforehand
        self.ipm = IPM(self.tf_buffer, self.camera_info, distortion=True)

        # We will now define the plane we want to project onto
        # The plane is defined by a normal and a distance to the origin
        # following the plane equation ax + by + cz + d = 0
        self.plane = Plane()
        self.plane.coef[2] = 1.0  # Normal in z direction

    def main(self):
        while rclpy.ok():
            # We will ask the user for a pixel to project
            point = Point2D(
                x = float(input('Enter pixel x: ')),
                y = float(input('Enter pixel y: '))
            )

            # Use the latest time for TF
            # Note that this is not the correct way to do this, but it is sufficient for this example
            # Normally, you would use the timestamp of the image/measurement you want to project
            # This is relevant as the tf tree is time-dependent and might change over time as the robot moves
            # This can lead to wrong projections, especially close to the horizon where the projection is very sensitive
            time = Time()

            # We will now project the pixel onto the plane using our library
            try:
                point = self.ipm.map_point(
                    self.plane,
                    point,
                    time,
                    plane_frame_id='map', # We defined a transform from the map to the camera earlier
                    output_frame_id='map' # We want the output to be in the same frame as the plane
                )

                # Print the result
                print(f'Projected point: {point.point.x}, {point.point.y}, {point.point.z}')

                # Now we will publish the projected point on a topic so we can visualize it in RViz
                self.point_pub.publish(point)
            except NoIntersectionError:
                print('No intersection found')


if __name__ == '__main__':
    rclpy.init()
    ipm_example = IPMExample()
    ipm_example.main()
    rclpy.shutdown()