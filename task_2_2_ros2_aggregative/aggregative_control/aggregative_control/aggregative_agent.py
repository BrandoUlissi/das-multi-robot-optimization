# ROS agent node for the aggretative task (2.2) of the DAS project
# Authors: Group 23 - Niccolò Antolini, Daniele Crivellari, Brando Ulissi

# Import the necessary libraries and parameters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray as MsgFloat
from time import sleep
from std_msgs.msg import Int64MultiArray
from std_msgs.msg import Int64


def get_grad_tracking(z_i, s_i, gamma):
    """
    Compute gradient tracking term for consensus-based optimization.
    
    Args:
        z_i: Current state of agent i
        s_i: Auxiliary tracking variable for agent i
        gamma: Tracking parameter
        
    Returns:
        Gradient tracking term: -2 * gamma * (z_i - s_i)
    """
    return -2 * gamma * (z_i - s_i)


class AggregativeAgent(Node):
    """
    Distributed Autonomous System Agent Node.
    
    This node implements a distributed optimization algorithm where multiple agents
    collaborate to solve an aggregative optimization problem while avoiding collisions.
    Each agent minimizes its local cost function while reaching consensus on a shared
    aggregative variable and maintaining safe distances from neighbors.
    """
    
    def __init__(self):
        super().__init__(
            "aggregative_control_agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        
        # Synchronization variables
        self.current_tick = -1
        
        # QoS profile for reliable synchronization communication
        sync_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Subscribe to synchronization tick for coordinated execution
        self.create_subscription(Int64, '/sync_tick', self.sync_callback, sync_qos)

        # Get algorithm parameters from launch file
        self.agent_id = self.get_parameter("id").value                    # Agent unique identifier
        self.z_i = np.array(self.get_parameter("z_init").value)          # Initial state vector
        self.t_i = np.array(self.get_parameter("target").value)          # Target/reference vector
        
        # Algorithm parameters
        self.alpha = self.get_parameter("alpha").value                   # Step size for state update
        self.gamma = self.get_parameter("gamma").value                   # Consensus tracking parameter
        self.delta = self.get_parameter("delta").value                   # Collision avoidance radius
        self.beta = self.get_parameter("beta").value                     # Repulsive force weight
        agent_ids = self.get_parameter("agent_ids").value                # List of all agent IDs in the system
        self.max_iters = self.get_parameter("maxT").value                # Maximum number of iterations
        
        # Algorithm state variables
        self.k = 0                                                       # Current iteration counter
        self.phase = 0                                                   # Current phase (0: publish, 1: compute)
        
        # Network topology variables
        self.neighbors = []                                              # List of neighboring agent IDs
        self.degree = 0                                                  # Number of neighbors (node degree)
        
        # Subscribe to neighbor list updates
        self.create_subscription(Int64MultiArray, f"/neighbors_{self.agent_id}", 
                               self.neighbors_callback, 10)
        
        # Dictionary to store received data from all neighbors
        # Key: agent_id, Value: dict with 'deg', 'z', 's', 'v' fields
        self.received_data = {}
        
        # Algorithm state variables
        self.s_i = np.copy(self.z_i)                                   # Initialize barycenter tracking variable
        self.v_i = get_grad_tracking(self.z_i, self.s_i, self.gamma)   # Initialize gradient tracking variable
        
        # Communication setup: subscribe to all agents and create publisher
        for i in agent_ids:
            self.create_subscription(MsgFloat, f"/agent_{i}", self.listener_callback, 10)
            self.received_data[i] = []
        
        self.publisher = self.create_publisher(MsgFloat, f"/agent_{self.agent_id}", 10)

        self.get_logger().info(f"Initialized AggregativeAgent with ID {self.agent_id}")


    def neighbors_callback(self, msg):
        """
        Callback for receiving neighbor list updates.
        Updates the current neighbor list and node degree.
        
        Args:
            msg: Int64MultiArray containing neighbor agent IDs
        """
        self.neighbors = list(msg.data)
        self.degree = len(self.neighbors)


    def sync_callback(self, msg):
        """
        Synchronization callback that triggers algorithm execution.
        Ensures all agents execute iterations in a coordinated manner.
        
        Args:
            msg: Int64 message containing the current synchronization tick
        """
        self.current_tick = msg.data
        if self.current_tick == self.k:
            self.timer_callback()


    def listener_callback(self, msg):
        """
        Callback for receiving data from neighboring agents.
        Parses and stores state information (z, s, v) from other agents.
        
        Message format: [agent_id, z_dim, degree, z_values..., s_values..., v_values...]
        
        Args:
            msg: Float64MultiArray containing neighbor's state data
        """
        try:
            data = msg.data
            if len(data) < 2:
                self.get_logger().warn("Received message too short")
                return
            
            # Parse message header
            agent_id = int(data[0])
            z_dim = int(data[1])
            
            # Validate message length
            expected_length = 3 + 3 * z_dim  # agent_id + z_dim + degree + z + s + v
            if len(data) != expected_length:
                self.get_logger().warn(f"Expected {expected_length} elements, got {len(data)}")
                return
            
            # Extract data fields
            deg_j = int(data[2])                                        # Neighbor's degree
            z_start = 3
            s_start = z_start + z_dim
            v_start = s_start + z_dim
            
            z_j = np.array(data[z_start:s_start])                      # Neighbor's state
            s_j = np.array(data[s_start:v_start])                      # Neighbor's consensus variable
            v_j = np.array(data[v_start:v_start + z_dim])              # Neighbor's tracking variable
            
            # Store received data
            if agent_id not in self.received_data:
                self.received_data[agent_id] = []
            
            self.received_data[agent_id] = {
                'deg': deg_j,
                'z': z_j,
                's': s_j,
                'v': v_j
            }

        except (IndexError, ValueError) as e:
            self.get_logger().warn(f"Failed to parse message: {e}")


    def publish_data(self):
        """
        Publish current agent's state data to all other agents.
        
        Message format: [agent_id, z_dim, degree, z_values..., s_values..., v_values...]
        """
        z_dim = len(self.z_i)
        
        msg = MsgFloat()
        msg.data = [float(self.agent_id), float(z_dim), float(self.degree)]
        msg.data.extend(self.z_i.tolist())
        msg.data.extend(self.s_i.tolist())
        msg.data.extend(self.v_i.tolist())
        
        self.publisher.publish(msg)


    def timer_callback(self):
        """
        Main algorithm execution callback.
        Implements a two-phase distributed optimization algorithm:
        
        Phase 0: Publish current state data to neighbors
        Phase 1: Receive neighbor data and compute state updates
        
        The algorithm alternates between these phases to ensure proper
        synchronization and data exchange between agents.
        """

        # Initial iteration: publish data and increment counter
        if self.k == 0:
            self.publish_data()
            self.get_logger().info(f"[{self.k}] Agent {self.agent_id}: z_i = {self.z_i}")
            self.k += 1
            return
        
        # PHASE 0: Update neighbor information and publish state
        if self.phase == 0:
            # Update degree based on current neighbors
            self.degree = len(self.neighbors)
            
            # Publish current state with updated degree
            self.publish_data()
            self.get_logger().info(f"[{self.k}] Phase 0 - Agent {self.agent_id}: Published with degree = {self.degree}")
            
            # Move to computation phase
            self.phase = 1
            self.k += 1
            return
                
        # PHASE 1: Computation phase - update agent state
        elif self.phase == 1:
            # Handle isolated agents (no neighbors)
            if len(self.neighbors) == 0:
                self.get_logger().info(f"[{self.k}] Agent {self.agent_id}: No neighbors, using local values only")
                
                # Compute local gradient (tracking + local cost)
                grad = 2 * (self.z_i - self.t_i) + 2 * self.gamma * (self.z_i - self.s_i)
                update = grad + self.v_i
                z_new = self.z_i - self.alpha * update
                
                # Update auxiliary variables without neighbors
                s_new = self.s_i + (z_new - self.z_i)
                grad_new = get_grad_tracking(z_new, s_new, self.gamma)
                grad_old = get_grad_tracking(self.z_i, self.s_i, self.gamma)
                v_new = self.v_i + grad_new - grad_old
                
                # Commit updates
                self.z_i = z_new
                self.s_i = s_new
                self.v_i = v_new
                
                self.publish_data()
                self.get_logger().info(f"[{self.k}] Agent {self.agent_id}: z_i = {self.z_i}")

                self.k += 1
                self.phase = 0

                # Check termination condition
                if self.k >= 2 * self.max_iters:
                    # Wait briefly before shutdown to ensure all messages are sent
                    self.get_logger().info(f"{self.max_iters} iterations reached, shutting down agent {self.agent_id}")
                    raise SystemExit
                return
            
            # Check if all neighbor messages have been received
            all_received = all(j in self.received_data and isinstance(self.received_data[j], dict) 
                             for j in self.neighbors)
            
            if not all_received:
                missing_neighbors = [j for j in self.neighbors 
                                   if j not in self.received_data or len(self.received_data[j]) == 0]
                self.get_logger().info(f"Agent {self.agent_id}: Not all messages received at iteration {self.k}, "
                                     f"missing neighbors: {missing_neighbors}")
                self.publish_data()
                self.k += 1
                self.phase = 0

                # Check termination condition
                if self.k >= 2 * self.max_iters:
                    # Wait briefly before shutdown to ensure all messages are sent
                    self.get_logger().info(f"{self.max_iters} iterations reached, shutting down agent {self.agent_id}")
                    raise SystemExit

                return

            # Extract neighbor data for computation
            neighbor_data = {}
            for j in self.neighbors:
                neighbor_data[j] = self.received_data[j]

            # Compute consensus weights using Metropolis-Hastings rule
            W = {}  # Weights for neighbors
            d_i = self.degree  # Current agent's degree

            for neighbor_id in self.neighbors:
                d_j = neighbor_data[neighbor_id]['deg']  # Neighbor's degree
                W[neighbor_id] = 1.0 / (1 + max(d_i, d_j))  # Metropolis-Hastings weight

            W_self = 1.0 - sum(W.values())  # Self-weight (diagonal element)

            # Compute state update
            # Local cost gradient: 2 * (z_i - t_i)
            # Tracking gradient: 2 * gamma * (z_i - s_i)
            grad = 2 * (self.z_i - self.t_i) + 2 * self.gamma * (self.z_i - self.s_i)

            # Add collision avoidance forces
            for j in self.neighbors:
                diff = self.z_i - neighbor_data[j]['z']
                dist = np.linalg.norm(diff)
                hij = max(0, self.delta - dist)  # Collision indicator function
                
                # Add repulsive force if agents are too close
                if dist > 0:  # Avoid division by zero
                    grad += (-4 * self.beta * hij / dist) * diff

            # Complete gradient update with tracking variable
            update = grad + self.v_i
            z_new = self.z_i - self.alpha * update

            # Update consensus variable s_i using weighted average
            s_new = self.s_i * W_self  # Self contribution
            for j in self.neighbors:
                s_new += W[j] * neighbor_data[j]['s']  # Neighbor contributions
            s_new += (z_new - self.z_i)  # Add state change

            # Update gradient tracking variable v_i
            grad_new = get_grad_tracking(z_new, s_new, self.gamma)
            grad_old = get_grad_tracking(self.z_i, self.s_i, self.gamma)

            v_new = self.v_i * W_self  # Self contribution
            for j in self.neighbors:
                v_new += W[j] * neighbor_data[j]['v']  # Neighbor contributions
            v_new += grad_new - grad_old  # Add gradient change

            self.get_logger().info(f"[{self.k}] Phase 1 - Agent {self.agent_id}: z_i = {self.z_i}")
            
            
            # Commit all updates
            self.z_i = z_new
            self.s_i = s_new
            self.v_i = v_new
                        

            # Publish updated state
            self.publish_data()

            # Move to next iteration
            self.k += 1
            self.phase = 0

            # Check termination condition
            if self.k >= 2 * self.max_iters:
                # Wait briefly before shutdown to ensure all messages are sent
                sleep(2)
                self.get_logger().info(f"{self.max_iters} iterations reached, shutting down agent {self.agent_id}")
                raise SystemExit

def main(args=None):
    """
    Main function to initialize and run the DAS agent node.
    
    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = AggregativeAgent()
    node.get_logger().info("Waiting for sync...")
    sleep(1)
    node.get_logger().info("Starting iterations")
    
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("AggregativeControl").info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()