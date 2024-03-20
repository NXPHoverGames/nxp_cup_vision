from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'debug', default_value='True',
            description='Enable debug mode for nxp_track_vision_node'),

        Node(
            package='nxp_cup_vision',
            executable='nxp_track_vision',
            name='nxp_track_vision_node',
            output='screen',
            parameters=[{'debug': LaunchConfiguration('debug')}],  # Use LaunchConfiguration for the parameter
            arguments=[
                # List any arguments your node needs here
            ],
        ),
    ])

