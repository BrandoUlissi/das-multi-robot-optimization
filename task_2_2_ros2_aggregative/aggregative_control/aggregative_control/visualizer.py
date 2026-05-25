# ROS visalization node for the aggretative task (2.2) of the DAS project
# Authors: Group 23 - Niccolò Antolini, Daniele Crivellari, Brando Ulissi

# Import the necessary libraries and parameters
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import csv
import re
import matplotlib.pyplot as plt


class Visualizer(Node):
    """
    ROS2 node for visualizing distributed autonomous systems from CSV trajectory data.

    This node visualizes the behavior of distributed autonomous agents by reading
    trajectory data from CSV files and publishing RViz markers. It supports:
    - Agent position visualization with 3D models
    - Communication range visualization
    - Target position markers
    - Agent trajectory trails
    - Network connectivity visualization
    - Convergence analysis plotting
    """
    
    def __init__(self):
        """Initialize the CSV Visualizer node with parameters and publishers."""
        super().__init__(
            "csv_visualizer",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )
        
        # Load ROS parameters
        self.csv_path = self.get_parameter("csv_path").value
        self.publish_rate = self.get_parameter("publish_rate").value
        self.target_path = self.get_parameter("target_path").value
        self.stl_path = self.get_parameter("stl_path").value
        self.r_comm = self.get_parameter("r_comm").value
        
        # Create publisher for visualization markers
        self.marker_pub = self.create_publisher(MarkerArray, "/agent_markers", 10)
        
        # Load and process CSV data
        all_positions, cost_values, gradient_values = self.load_csv_data(self.csv_path)
        self.positions_per_tick = all_positions
        
        # Generate convergence plots if data is available
        if len(cost_values) > 0 and len(gradient_values) > 0:
            self.plot_convergence_analysis(cost_values, gradient_values)
        
        # Initialize system parameters
        self.num_ticks = len(self.positions_per_tick)
        self.num_agents = len(self.positions_per_tick[0]) if self.positions_per_tick else 0
        self.current_tick = 0
        
        # Load target positions
        self.target_positions = self.load_target_positions(self.target_path)
        
        # Generate unique colors for each agent
        self.agent_colors = self.generate_agent_colors(self.num_agents)
        
        # Initialize trail markers for each agent
        self.trail_markers = [self.create_trail_marker(i) for i in range(self.num_agents)]
        
        # Log initialization status
        self.get_logger().info(
            f"Initialized CSV Visualizer: {self.num_ticks} ticks, "
            f"{self.num_agents} agents from {self.csv_path}"
        )
        
        # Create timer for periodic publishing
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_visualization_markers)

    def load_csv_data(self, csv_file_path):
        """
        Load agent trajectory data from CSV file.
        
        Expected CSV format:
        - Columns named as 'z{agent_id}_{dimension}' (e.g., z0_1, z0_2 for agent 0)
        - Last two columns contain cost and gradient values
        
        Args:
            csv_file_path (str): Path to the CSV file containing trajectory data
            
        Returns:
            tuple: (all_positions, cost_values, gradient_values)
                - all_positions: List of position data for each time tick
                - cost_values: Array of cost function values
                - gradient_values: Array of gradient norm values
        """
        all_positions = []
        cost_values = []
        gradient_values = []
        
        try:
            with open(csv_file_path, newline="") as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                
                # Parse column headers to map agent coordinates
                coordinate_pattern = re.compile(r"z(\d+)_([12])")
                agent_coordinate_map = {}
                
                # Build mapping of agent IDs to their coordinate column indices
                for column_idx, column_name in enumerate(header):
                    match = coordinate_pattern.fullmatch(column_name)
                    if match:
                        agent_id = int(match.group(1))
                        dimension = int(match.group(2))  # 1=x, 2=y
                        
                        if agent_id not in agent_coordinate_map:
                            agent_coordinate_map[agent_id] = [None, None]
                        agent_coordinate_map[agent_id][dimension - 1] = column_idx
                
                # Sort agents by ID and extract coordinate indices
                sorted_agents = sorted(agent_coordinate_map.items())
                coordinate_indices = [tuple(coord_pair) for _, coord_pair in sorted_agents]
                
                # Process each data row
                for row in reader:
                    tick_positions = []
                    
                    # Extract agent positions for this time tick
                    for x_idx, y_idx in coordinate_indices:
                        x_coord = float(row[x_idx])
                        y_coord = float(row[y_idx])
                        tick_positions.append((x_coord, y_coord))
                    
                    all_positions.append(tick_positions)
                    
                    # Extract cost and gradient values (last two columns)
                    cost_values.append(float(row[-2]))
                    gradient_values.append(float(row[-1]))
                
                return all_positions, np.array(cost_values), np.array(gradient_values)
                
        except Exception as error:
            self.get_logger().error(f"Failed to load CSV data from {csv_file_path}: {error}")
            return [], np.array([]), np.array([])

    def load_target_positions(self, target_file_path):
        """
        Load target positions from file.
        
        Expected format: Each line contains "label: [x, y]"
        
        Args:
            target_file_path (str): Path to target positions file
            
        Returns:
            list: List of target position tuples (x, y)
        """
        target_positions = []
        
        try:
            with open(target_file_path, "r") as file:
                for line in file:
                    if ":" in line:
                        parts = line.strip().split(":")
                        coordinates = eval(parts[1])  # Parse coordinate list
                        target_positions.append(tuple(coordinates))
                        
        except Exception as error:
            self.get_logger().error(f"Failed to load target positions from {target_file_path}: {error}")
            
        return target_positions

    def plot_convergence_analysis(self, cost_values, gradient_values):
        """
        Generate convergence analysis plots for the optimization process.
        
        Creates two subplots:
        1. Cost function evolution over iterations
        2. Gradient norm evolution over iterations
        
        Args:
            cost_values (np.array): Array of cost function values
            gradient_values (np.array): Array of gradient norm values
        """
        if len(cost_values) < 4 or len(gradient_values) < 4:
            self.get_logger().warn("Insufficient data for convergence analysis")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Convergence Analysis - Distributed Optimization", 
                    fontsize=16, fontweight='bold')
        
        # Plot cost function evolution (skip first 3 points for stability)
        axes[0].semilogy(np.abs(cost_values[3:]))
        axes[0].set_ylabel(r"$\ell(z)$")
        axes[0].set_xlabel(r"Iteration $k$")
        axes[0].grid(True, alpha=0.3)
        
        # Plot gradient norm evolution
        axes[1].semilogy(gradient_values[3:])
        axes[1].set_xlabel(r"Iteration $k$")
        axes[1].set_ylabel(r"$\||\nabla \ell(z)\||$")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def generate_agent_colors(self, num_agents):
        """
        Generate distinct colors for each agent using matplotlib colormap.
        
        Args:
            num_agents (int): Number of agents requiring unique colors
            
        Returns:
            list: List of RGBA color tuples for each agent
        """
        colormap = plt.get_cmap('tab10')
        return [colormap(agent_id % 10) for agent_id in range(num_agents)]

    def publish_visualization_markers(self):
        """
        Timer callback to publish visualization markers for current time tick.
        
        Publishes:
        - Agent position markers with 3D models
        - Agent text labels
        - Communication range circles
        - Agent trajectory trails
        - Target position markers
        - Network connectivity lines
        """
        # Check if simulation is complete
        if self.current_tick >= self.num_ticks:
            self.get_logger().info("Simulation complete. All ticks published.")
            rclpy.shutdown()
            return
        
        current_positions = self.positions_per_tick[self.current_tick]
        self.get_logger().info(f"Publishing tick {self.current_tick + 1}/{self.num_ticks}")
        
        marker_array = MarkerArray()
        
        # Create markers for each agent
        for agent_id, (x_pos, y_pos) in enumerate(current_positions):
            agent_color = self.agent_colors[agent_id]
            
            # Add agent 3D model marker
            marker_array.markers.append(
                self.create_agent_marker(agent_id, x_pos, y_pos, agent_color)
            )
            
            # Add agent text label
            marker_array.markers.append(
                self.create_text_marker(agent_id, x_pos, y_pos, f"Agent_{agent_id}")
            )
            
            # Add communication range visualization
            marker_array.markers.append(
                self.create_communication_range_marker(agent_id, x_pos, y_pos, agent_color)
            )
            
            # Update and add trajectory trail
            self.trail_markers[agent_id].points.append(self.create_point(x_pos, y_pos))
            self.trail_markers[agent_id].header.stamp = self.get_clock().now().to_msg()
            marker_array.markers.append(self.trail_markers[agent_id])
        
        # Add target position markers
        for target_id, (x_pos, y_pos) in enumerate(self.target_positions):
            if target_id < len(self.agent_colors):  # Ensure color availability
                target_color = self.agent_colors[target_id]
                
                marker_array.markers.append(
                    self.create_target_marker(target_id, x_pos, y_pos, target_color)
                )
                marker_array.markers.append(
                    self.create_text_marker(target_id + 500, x_pos, y_pos, f"Target_{target_id}")
                )
        
        # Add network connectivity visualization
        marker_array.markers.append(
            self.create_network_connectivity_marker(current_positions)
        )
        
        # Publish all markers
        self.marker_pub.publish(marker_array)
        self.current_tick += 1

    def create_agent_marker(self, agent_id, x_pos, y_pos, color):
        """
        Create a 3D mesh marker for an agent.
        
        Args:
            agent_id (int): Unique identifier for the agent
            x_pos (float): X coordinate position
            y_pos (float): Y coordinate position
            color (tuple): RGBA color tuple
            
        Returns:
            Marker: RViz marker for agent visualization
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "agents"
        marker.id = agent_id
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = self.stl_path
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = x_pos
        marker.pose.position.y = y_pos
        marker.pose.position.z = 0.0
        
        # Set orientation (90-degree rotation around X-axis)
        marker.pose.orientation.x = 0.7071
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 0.7071
        
        # Set scale
        marker.scale.x = 0.0012
        marker.scale.y = 0.0012
        marker.scale.z = 0.0012
        
        # Set color
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        
        return marker

    def create_target_marker(self, target_id, x_pos, y_pos, color):
        """
        Create a sphere marker for target positions.
        
        Args:
            target_id (int): Unique identifier for the target
            x_pos (float): X coordinate position
            y_pos (float): Y coordinate position
            color (tuple): RGBA color tuple
            
        Returns:
            Marker: RViz marker for target visualization
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "targets"
        marker.id = 100 + target_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = x_pos
        marker.pose.position.y = y_pos
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set scale (sphere diameter)
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        
        # Set color
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        
        return marker

    def create_text_marker(self, marker_id, x_pos, y_pos, label_text):
        """
        Create a text marker for labeling agents and targets.
        
        Args:
            marker_id (int): Unique marker identifier
            x_pos (float): X coordinate position
            y_pos (float): Y coordinate position
            label_text (str): Text to display
            
        Returns:
            Marker: RViz text marker
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "labels"
        marker.id = marker_id + 1000
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Set position (slightly above the object)
        marker.pose.position.x = x_pos
        marker.pose.position.y = y_pos
        marker.pose.position.z = 0.4
        marker.pose.orientation.w = 1.0
        
        # Set text properties
        marker.scale.z = 0.2  # Text height
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.text = label_text
        
        return marker

    def create_communication_range_marker(self, agent_id, x_pos, y_pos, color):
        """
        Create a cylindrical marker to visualize communication range.
        
        Args:
            agent_id (int): Agent identifier
            x_pos (float): X coordinate position
            y_pos (float): Y coordinate position
            color (tuple): RGBA color tuple
            
        Returns:
            Marker: RViz cylinder marker for communication range
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "communication_range"
        marker.id = agent_id + 2000
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = x_pos
        marker.pose.position.y = y_pos
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set scale (cylinder dimensions)
        marker.scale.x = 2 * self.r_comm  # Diameter in X
        marker.scale.y = 2 * self.r_comm  # Diameter in Y
        marker.scale.z = 0.01        # Very thin cylinder
        
        # Set semi-transparent color
        marker.color.a = 0.2
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        
        return marker

    def create_network_connectivity_marker(self, agent_positions):
        """
        Create line markers to visualize network connectivity between agents.
        
        Draws lines between agents that are within communication range of each other.
        
        Args:
            agent_positions (list): List of (x, y) tuples for agent positions
            
        Returns:
            Marker: RViz line list marker showing network connections
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "network_edges"
        marker.id = 9999
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Set line properties
        marker.scale.x = 0.02  # Line width
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        
        # Find and add connections between nearby agents
        marker.points = []
        for i, (x1, y1) in enumerate(agent_positions):
            for j, (x2, y2) in enumerate(agent_positions):
                if j <= i:  # Avoid duplicate connections
                    continue
                    
                # Calculate distance between agents
                distance = np.hypot(x2 - x1, y2 - y1)
                
                # Add connection if within communication range
                if distance < self.r_comm:
                    point1 = Point(x=x1, y=y1, z=0.05)
                    point2 = Point(x=x2, y=y2, z=0.05)
                    marker.points.extend([point1, point2])
        
        return marker

    def create_trail_marker(self, agent_id):
        """
        Create a line strip marker for agent trajectory trails.
        
        Args:
            agent_id (int): Agent identifier
            
        Returns:
            Marker: RViz line strip marker for trajectory visualization
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.ns = "trajectories"
        marker.id = 3000 + agent_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Set line properties
        marker.scale.x = 0.05  # Line width
        marker.color.a = 0.8   # Semi-transparent
        
        # Set agent-specific color
        agent_color = self.agent_colors[agent_id]
        marker.color.r = agent_color[0]
        marker.color.g = agent_color[1]
        marker.color.b = agent_color[2]
        
        marker.pose.orientation.w = 1.0
        marker.points = []  # Will be populated during simulation
        
        return marker

    def create_point(self, x_coord, y_coord, z_coord=0.0):
        """
        Create a geometry_msgs Point.
        
        Args:
            x_coord (float): X coordinate
            y_coord (float): Y coordinate
            z_coord (float): Z coordinate (default: 0.0)
            
        Returns:
            Point: geometry_msgs Point object
        """
        point = Point()
        point.x = x_coord
        point.y = y_coord
        point.z = z_coord
        return point


def main(args=None):
    """
    Main function to initialize and run the Visualizer node.
    
    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    
    try:
        node = Visualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()