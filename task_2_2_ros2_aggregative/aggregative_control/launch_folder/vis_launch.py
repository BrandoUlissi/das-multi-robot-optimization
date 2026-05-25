# ROS launch node for the visualization of aggretative task (2.2) of the DAS project
# Authors: Group 23 - Niccolò Antolini, Daniele Crivellari, Brando Ulissi

# Import the necessary libraries and parameters
from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

PATH_TO_PACKAGE = get_package_share_directory('aggregative_control')


def generate_launch_description():
    """
    Generate launch description for DAS (Distributed Aggregative Systems) visualization.
    
    This launch file starts the visualization system for monitoring and displaying
    the behavior of the distributed aggregative control algorithm. It includes:
    
    1. Visualizer Node: Reads logged data and publishes visualization markers
    2. RViz2 Node: Provides 3D visualization interface
    3. Static Transform Publisher: Sets up coordinate frame relationships
    
    Returns:
        LaunchDescription: Complete launch configuration for the visualization system
    """
    
    package_name = "aggregative_control"

    r_comm = 3.5                      # Communication radius


    # Visualization Node
    # Reads agent trajectory data from CSV log and target positions from text file
    # Plots cost and gradient and publishes RViz markers for real-time visualization of agent movements
    visualizer_node = Node(
        package=package_name,
        executable="visualizer",                           # Executable name for visualizer
        name="visualizer",                                 # Node name
        parameters=[
            {
                # Path to CSV file containing logged agent positions and system metrics
                "csv_path": os.path.join(PATH_TO_PACKAGE, "data", "agent_data_log.csv"),
                
                # Path to text file containing target positions for each agent
                "target_path": os.path.join(PATH_TO_PACKAGE, "data", "target_positions.txt"),
                
                # Path to 3D model file for agent representation (quadcopter STL model)
                "stl_path": "package://aggregative_control/Quadcopter_model.stl",
                
                # Publishing rate for visualization updates (Hz)
                "publish_rate": 3.0,

                # Communication radius for agent interactions
                "r_comm": r_comm,  
            }
        ],
        output="screen",                                # Display node output in terminal
    )

    # RViz2 Visualization Node
    # Provides 3D graphical interface for visualizing the DAS algorithm execution
    # Uses a pre-configured RViz configuration file for optimal display settings
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",                             # RViz2 executable
        name="rviz2",                                   # Node name
        arguments=[
            "-d",                                       # Configuration file flag
            os.path.join(PATH_TO_PACKAGE, "config", "aggregative_view.rviz")  # Path to RViz config
        ],
        output="screen",                                # Display RViz output in terminal
    )

    # Static Transform Publisher
    # Establishes the coordinate frame transformation between 'world' and 'map' frames
    # This is required for proper visualization in RViz and ensures all markers
    # are displayed in the correct coordinate system
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",        # TF2 static publisher executable
        name="static_tf_pub_world",                     # Node name
        arguments=[
            # Translation parameters (x, y, z in meters)
            "--x", "0", "--y", "0", "--z", "0",
            
            # Rotation parameters (roll, pitch, yaw in radians)
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            
            # Frame relationship: world -> map (identity transformation)
            "--frame-id", "world", 
            "--child-frame-id", "map"
        ],
        output="screen"                                 # Display transform info in terminal
    )

    # Return complete launch description with all visualization components
    return LaunchDescription([
        visualizer_node,    # CSV data reader and marker publisher
        rviz_node,         # 3D visualization interface
        static_tf_node     # Coordinate frame publisher
    ])