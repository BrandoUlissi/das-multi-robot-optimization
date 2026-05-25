# ROS supervisor node for the aggretative task (2.2) of the DAS project
# Authors: Group 23 - Niccolò Antolini, Daniele Crivellari, Brando Ulissi

# Import the necessary libraries and parameters
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray as MsgFloat
from std_msgs.msg import Int64MultiArray, Int64
import numpy as np
import pandas as pd
import os


class NeighborSupervisor(Node):
    """
    Neighbor Supervisor Node for Distributed Aggregative Systems (DAS).
    
    This node acts as a centralized coordinator that:
    1. Monitors agent positions and states
    2. Computes and publishes neighbor relationships based on communication range
    3. Synchronizes algorithm execution across all agents
    4. Logs system performance metrics (cost, gradients) to CSV
    5. Manages algorithm termination after maximum iterations
    """
    
    def __init__(self):
        super().__init__("neighbor_supervisor",
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True
        )

        # Get system parameters from launch file
        self.agent_ids = self.get_parameter("agent_ids").value                    # List of all agent IDs
        communication_time = self.get_parameter("communication_time").value      # Timer period for synchronization
        self.max_iters = self.get_parameter("max_iters").value                   # Maximum logical iterations
        targets = self.get_parameter("targets").value                            # Flattened target positions
        self.r_comm = self.get_parameter("r_comm").value                         # Communication range
        self.gamma = self.get_parameter("gamma").value                           # Consensus parameter
        self.delta = self.get_parameter("delta").value                           # Collision avoidance radius
        self.beta = self.get_parameter("beta").value                             # Repulsive force gain
        initial_positions_param = self.get_parameter("initial_positions").value  # Flattened initial positions

        self.get_logger().info(f"Initializing NeighborSupervisor with agent_ids: {self.agent_ids}")

        # Parse and store target positions
        self.targets = {}
        self._parse_and_save_targets(targets)

        # Initialize agent positions and neighbor relationships
        self.positions = {}
        self.neighbors = {}
        self._parse_initial_positions(initial_positions_param)

        # Set up communication: subscribers for agent data, publishers for neighbor lists
        self.subscribers = []
        self.agent_publishers = {}
        self.received_this_tick = set()  # Track which agents have published in current tick
        self._setup_communication()

        # Initialize CSV logging for performance metrics
        self._setup_csv_logging()

        # Create timer for periodic neighbor calculation and synchronization
        self.timer = self.create_timer(communication_time, self.timer_callback)

        # Calculate initial neighbors if all positions are available
        if all(pos is not None for pos in self.positions.values()):
            self.calculate_and_publish_neighbors()

        # Set up synchronization publisher with reliable QoS
        self._setup_synchronization()

        self.get_logger().info("NeighborSupervisor initialization complete")


    def _parse_and_save_targets(self, targets):
        """
        Parse target positions from flattened parameter and save to file.
        
        Args:
            targets: Flattened list of target coordinates [x1, y1, x2, y2, ...]
        """
        targets_filename = "target_positions.txt"
        with open(targets_filename, 'w') as f:
            for i, agent_id in enumerate(self.agent_ids):
                target_start_idx = i * 2
                target_end_idx = target_start_idx + 2
                if target_end_idx <= len(targets):
                    target = targets[target_start_idx:target_end_idx]
                    self.targets[agent_id] = np.array(target)
                    f.write(f"Agent {agent_id}: [{target[0]}, {target[1]}]\n")
                else:
                    f.write(f"Agent {agent_id}: No target available\n")
        self.get_logger().info(f"Target positions saved to {targets_filename}")


    def _parse_initial_positions(self, initial_positions_param):
        """
        Parse initial positions from flattened parameter.
        
        Args:
            initial_positions_param: Flattened list of initial coordinates [x1, y1, x2, y2, ...]
        """
        for i, agent_id in enumerate(self.agent_ids):
            pos_start_idx = i * 2
            pos_end_idx = pos_start_idx + 2
            if pos_end_idx <= len(initial_positions_param):
                self.positions[agent_id] = np.array(initial_positions_param[pos_start_idx:pos_end_idx])
            else:
                self.positions[agent_id] = None


    def _setup_communication(self):
        """
        Set up subscribers to receive agent data and publishers to send neighbor lists.
        """
        for i in self.agent_ids:
            # Subscribe to each agent's state data
            topic_name = f"/agent_{i}"
            sub = self.create_subscription(MsgFloat, topic_name, self.make_callback(i), 10)
            self.subscribers.append(sub)

            # Create publisher for each agent's neighbor list
            pub_topic_name = f"/neighbors_{i}"
            self.agent_publishers[i] = self.create_publisher(Int64MultiArray, pub_topic_name, 10)


    def _setup_csv_logging(self):
        """
        Initialize CSV file for logging system performance metrics.
        """
        # Create column names: z coordinates for each agent + system metrics
        columns = []
        for agent_id in self.agent_ids:
            columns.append(f"z{agent_id}_1")  # x-coordinate
            columns.append(f"z{agent_id}_2")  # y-coordinate
        columns.extend(["cost", "grad_tot"])  # System cost and total gradient norm

        # Initialize CSV file (remove if exists)
        self.csv_filename = "agent_data_log.csv"
        if os.path.exists(self.csv_filename):
            os.remove(self.csv_filename)

        df = pd.DataFrame(columns=columns)
        df.to_csv(self.csv_filename, index=False)


    def _setup_synchronization(self):
        """
        Set up synchronization publisher for coordinating agent execution.
        """
        sync_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.sync_publisher = self.create_publisher(Int64, '/sync_tick', sync_qos)
        self.sync_tick = 0
        
        # Publish initial sync tick
        msg = Int64()
        msg.data = self.sync_tick
        self.sync_publisher.publish(msg)


    def make_callback(self, agent_id):
        """
        Create a callback function for receiving data from a specific agent.
        
        Args:
            agent_id: ID of the agent for which to create the callback
            
        Returns:
            Callback function that processes messages from the specified agent
        """
        def callback(msg):
            try:
                data = msg.data
                if len(data) < 2:
                    self.get_logger().warn(f"Received message from agent {agent_id} too short")
                    return

                # Parse message header
                received_agent_id = int(data[0])
                z_dim = int(data[1])

                # Validate message length
                expected_length = 3 + 3 * z_dim  # agent_id + z_dim + degree + z + s + v
                if len(data) != expected_length:
                    self.get_logger().warn(f"Expected {expected_length} elements from agent {agent_id}, got {len(data)}")
                    return

                # Extract position data (z values)
                z_start = 3
                z_values = np.array(data[z_start:z_start + z_dim])

                # Update agent position
                self.positions[received_agent_id] = z_values
                self.received_this_tick.add(received_agent_id)

                self.get_logger().debug(f"Updated data for agent {received_agent_id}: z={z_values}")

            except (IndexError, ValueError) as e:
                self.get_logger().warn(f"Failed to parse message from agent {agent_id}: {e}")

        return callback


    def calculate_and_publish_neighbors(self):
        """
        Calculate neighbor relationships based on communication range and publish to agents.
        
        Two agents are neighbors if the Euclidean distance between them is less than r_comm.
        """
        for i in self.agent_ids:
            neighbors = []
            pos_i = self.positions[i]
            
            if pos_i is None:
                self.get_logger().warn(f"Position for agent {i} is None, skipping neighbor calculation")
                continue

            # Find all agents within communication range
            for j in self.agent_ids:
                if i == j:  # Skip self
                    continue
                    
                pos_j = self.positions[j]
                if pos_j is None:
                    continue

                distance = np.linalg.norm(pos_i - pos_j)
                if distance < self.r_comm:
                    neighbors.append(j)

            # Store and publish neighbor list
            self.neighbors[i] = neighbors

            msg = Int64MultiArray()
            msg.data = neighbors
            self.agent_publishers[i].publish(msg)
            self.get_logger().debug(f"Agent {i} neighbors: {neighbors}")


    def timer_callback(self):
        """
        Main timer callback for system coordination and monitoring.
        
        Handles:
        1. Neighbor calculation and publishing
        2. Two-phase synchronization (Phase 0: Publish, Phase 1: Compute)
        3. Performance logging during computation phase
        4. Algorithm termination after maximum iterations
        """
        # Determine current phase and logical iteration
        current_phase = self.sync_tick % 2  # 0 = publish phase, 1 = compute phase
        logical_iteration = self.sync_tick // 2

        # Check termination condition (maximum logical iterations reached)
        if logical_iteration >= self.max_iters:
            self.get_logger().info(f"Logical iteration {self.max_iters} reached. Shutting down node...")
            self.destroy_node()
            rclpy.shutdown()
            return
        
        # Check if all agent positions are available
        if any(pos is None for pos in self.positions.values()):
            self.get_logger().warn("Not all positions available yet")
            return

        # Calculate and publish neighbor relationships
        self.calculate_and_publish_neighbors()
        
        phase_name = "Phase 0 (Publish)" if current_phase == 0 else "Phase 1 (Compute)"
        self.get_logger().info(f"Sync tick {self.sync_tick} - Logical iter {logical_iteration} - {phase_name}")

        # Performance logging and monitoring during computation phase
        if current_phase == 1:  # Computation phase
            # Wait for all agents to publish their data
            if len(self.received_this_tick) < len(self.agent_ids):
                self.get_logger().info(
                    f"Waiting for all agents to publish. Received: {len(self.received_this_tick)}/{len(self.agent_ids)}"
                )
                return

            # Calculate system performance metrics
            self._calculate_and_log_metrics()

        # Increment sync tick and publish for next phase
        self.sync_tick += 1
        
        msg = Int64()
        msg.data = self.sync_tick
        self.sync_publisher.publish(msg)

        # Clear received data tracking for next tick
        self.received_this_tick.clear()


    def _calculate_and_log_metrics(self):
        """
        Calculate and log system performance metrics to CSV.
        
        Metrics include:
        - Individual agent positions
        - Total system cost (tracking + collision avoidance)
        - Total gradient norm
        """
        row = {}
        total_cost = 0
        grad = np.zeros((len(self.agent_ids), 2))

        # Calculate aggregative variable (mean position)
        sigma = np.mean(
            [self.positions[agent_id] for agent_id in self.agent_ids if self.positions[agent_id] is not None],
            axis=0
        )

        # Calculate metrics for each agent
        for agent_id in self.agent_ids:
            pos = self.positions[agent_id]
            
            # Log position coordinates
            if pos is not None:
                row[f"z{agent_id}_1"] = pos[0]
                row[f"z{agent_id}_2"] = pos[1]
            else:
                row[f"z{agent_id}_1"] = None
                row[f"z{agent_id}_2"] = None

            # Calculate local cost components
            # 1. Distance to target cost
            target_cost = np.linalg.norm(self.positions[agent_id] - self.targets[agent_id]) ** 2
            total_cost += target_cost
            
            # 2. Consensus cost (distance to mean)
            consensus_cost = self.gamma * np.linalg.norm(self.positions[agent_id] - sigma) ** 2
            total_cost += consensus_cost
            
            # Calculate local gradient components
            grad[agent_id] = 2 * (self.positions[agent_id] - self.targets[agent_id])  # Target gradient
            grad[agent_id] += 2 * self.gamma * (self.positions[agent_id] - sigma)     # Consensus gradient

            # 3. Collision avoidance cost and gradient
            for j in self.neighbors[agent_id]:
                diff = (self.positions[agent_id] - self.positions[j] 
                       if self.positions[j] is not None 
                       else np.zeros_like(self.positions[agent_id]))
                dist = np.linalg.norm(diff)
                hij = max(0, self.delta - dist)  # Collision indicator function
                
                # Add collision cost
                total_cost += self.beta * hij ** 2
                
                # Add repulsive gradient
                if dist > 0:  # Avoid division by zero
                    grad[agent_id] += (-4 * self.beta * hij / dist) * diff
                else:
                    grad[agent_id] += np.zeros_like(diff)

        # Store system-wide metrics
        row["cost"] = total_cost
        row["grad_tot"] = np.linalg.norm(grad)

        # Log to CSV file
        df = pd.DataFrame([row])
        df.to_csv(self.csv_filename, mode='a', header=False, index=False)


def main(args=None):
    """
    Main function to initialize and run the neighbor supervisor node.
    
    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = NeighborSupervisor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()