from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nxp_cup_vision',  # Replace with your package name
            executable='nxp_track_vision',  # Replace with the name of your node/executable
            name='nxp_track_vision_node',  # A name for the launched node
            output='screen',  # Choose where to display the node's output
            parameters=[{
                # List any ROS parameters here
            }],
            arguments=[
                # List any arguments your node needs here
            ],
        ),
    ])

