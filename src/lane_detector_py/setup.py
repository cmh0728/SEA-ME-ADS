from setuptools import setup

package_name = 'lane_detector_py'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/lane_detector.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Simple lane detection with rclpy + OpenCV',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lane_detector_node = lane_detector.lane_detector_node:main',
        ],
    },
)
