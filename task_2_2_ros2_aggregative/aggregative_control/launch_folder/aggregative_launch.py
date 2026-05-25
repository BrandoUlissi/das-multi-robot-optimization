# ROS launch node for the aggregative task (2.2) of the DAS project
# Authors: Group 23 - Niccolò Antolini, Daniele Crivellari, Brando Ulissi

# Import the necessary libraries and parameters
from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np

# Set random seed for consistent initial conditions across runs
np.random.seed(3)  # For reproducibility


def generate_launch_description():
    """
    Generate launch description for the aggregative control simulation.

    This launch file initializes the core computation nodes for the distributed
    aggregative control algorithm. It includes:

    1. Supervisor Node: Manages neighborhood updates and communication logic
    2. Agent Nodes: Implements distributed control logic for each agent

    Returns:
        LaunchDescription: Complete launch configuration for the system
    """

    # Simulation and algorithm parameters
    MAXITERS = 1000                   # Maximum number of iterations
    COMM_TIME = 0.2                   # Communication interval (seconds)
    N = 6                             # Number of agents
    d = 2                             # Space dimensionality (2D workspace)

    # Aggregative control algorithm parameters
    alpha = 0.02                      # Step size for gradient update
    gamma = 0.01                      # Parameter for dynamic update law
    delta = 1.0                       # Scaling factor in local cost function
    beta = 2.0                        # Weighting factor for aggregative term
    r_comm = 3.5                      # Communication radius

    # Generate random initial positions and target positions for each agent
    z_init = np.random.uniform(0, 10, size=(N, d))  # Initial states
    t_i = np.random.uniform(0, 10, size=(N, d))     # Target positions

    # Container for all nodes to be launched
    node_list = []
    package_name = "aggregative_control"  # Package containing ROS executables

    # Supervisor Node
    # Handles dynamic neighbor detection and shared communication logic
    supervisor_node = Node(
        package=package_name,
        executable="supervisor",                   # Supervisor executable
        name="supervisor",                         # Node name
        parameters=[
            {
                "agent_ids": list(range(N)),                      # IDs of all agents
                "initial_positions": z_init.flatten().tolist(),  # Initial positions (flattened)
                "communication_time": float(COMM_TIME),          # Communication interval
                "max_iters": int(MAXITERS),                      # Simulation duration
                "targets": t_i.flatten().tolist(),               # Target positions (flattened)
                "r_comm": float(r_comm),                         # Communication radius
                "gamma": float(gamma),                           # Algorithm parameter
                "delta": float(delta),                           # Algorithm parameter
                "beta": float(beta),                             # Algorithm parameter
            }
        ],
        output="screen"                           # Display node output in terminal
    )
    node_list.append(supervisor_node)

    # Agent Nodes
    # Each agent runs an instance of the distributed algorithm
    for i in range(N):
        node_list.append(
            Node(
                package=package_name,
                namespace=f"agent_{i}",                      # Unique namespace for each agent
                executable="agent",                          # Agent executable
                parameters=[
                    {
                        "id": int(i),                        # Agent ID
                        "z_init": [float(val) for val in z_init[i]],  # Initial position
                        "target": [float(val) for val in t_i[i]],     # Target position
                        "maxT": int(MAXITERS),               # Simulation time
                        "alpha": float(alpha),               # Algorithm parameter
                        "gamma": float(gamma),               # Algorithm parameter
                        "delta": float(delta),               # Algorithm parameter
                        "beta": float(beta),                 # Algorithm parameter
                        "agent_ids": list(range(N)),         # List of all agent IDs
                    }
                ],
                output="screen",                             # Display node output in terminal
                prefix=f'xterm -title "agent_{i}" -fg white -bg black -fs 12 -fa "Monospace" -hold -e',  # Launch each agent in a separate xterm
            )
        )

    # Return complete launch description with supervisor and agent nodes
    return LaunchDescription(node_list)
