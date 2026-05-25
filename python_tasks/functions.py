# Functions for solving distributed optimization problems
# Authors: Group 23 - Niccolò Antolini, Daniele Crivellari, Brando Ulissi

# Import the necessary libraries and parameters
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import *
import plots as plot


# Function to create different graph topologies for distributed algorithms
def create_graphs():
    """
    Create and return a list of different graph topologies for distributed algorithms.
    
    This function generates various network topologies commonly used in distributed optimization
    to test algorithm performance under different communication structures.

    Returns:
        list: A list containing the created graphs with different topologies:
              - Erdos-Renyi random graph
              - Star graph 
              - Ladder graph
              - Path graph
              - Cycle graph
              - Complete graph
              Shape: (6,).
    """
    erGraph = nx.erdos_renyi_graph(NN, p=0.4, seed=graph_seed)      # Erdos-Renyi random graph with connection probability 0.4
    starGraph = nx.star_graph(NN-1)                                 # Star graph with one central node connected to all others
    pathGraph = nx.path_graph(NN)                                   # Path graph where nodes form a linear chain
    cycleGraph = nx.cycle_graph(NN)                                 # Cycle graph where nodes form a closed ring
    completeGraph = nx.complete_graph(NN)                           # Complete graph where every node connects to every other
    ladderGraph = nx.ladder_graph(NN//2)                            # Ladder graph with parallel paths connected by rungs

    graphs = [erGraph, starGraph, ladderGraph, pathGraph, cycleGraph, completeGraph]  # List of all created graphs
    return graphs                                                   # Return list of graphs for algorithm testing


# Function to get adjacency and weight matrices for distributed consensus
def get_matrices(g, dim=NN):
    """
    Compute adjacency matrix and Metropolis-Hastings weight matrix for distributed consensus.
    
    The Metropolis-Hastings weights ensure doubly stochastic properties required for
    convergence in distributed consensus algorithms by balancing node degrees.
    
    Args:
        g (networkx.Graph): Input graph representing communication topology
        dim (int): Number of nodes in the graph (default: NN from config)
        
    Returns:
        tuple: (Adj, W) where:
               - Adj: Binary adjacency matrix. Shape: (dim, dim)
               - W: Doubly stochastic weight matrix using Metropolis-Hastings rule. Shape: (dim, dim)
    """
    # Get binary adjacency matrix from NetworkX graph
    Adj = nx.adjacency_matrix(g).toarray()
    
    # Check graph connectivity using matrix powers method
    # A graph is connected if (A + I)^n has all positive entries
    test = np.linalg.matrix_power(Adj + np.eye(dim), dim)
    if np.all(test <= 0):                                           # If any entry is non-positive...
        print("Graph is not connected")                             # ... warn about disconnected graph
    
    # Initialize weight matrix with zeros
    W = np.zeros((dim, dim))
    
    # Get degree of each node for Metropolis-Hastings weight computation
    degrees = dict(g.degree())
    
    # Compute Metropolis-Hastings weights for connected node pairs
    for i in range(dim):                                            # For each node i
        for j in range(dim):                                        # For each potential neighbor j
            if i != j and Adj[i,j] == 1:                           # If nodes i and j are connected (but not same node)
                # Metropolis-Hastings weight: 1/(1 + max(degree_i, degree_j))
                W[i,j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        
        # Set diagonal weight to ensure each row sums to 1 (doubly stochastic property)
        W[i,i] = 1.0 - np.sum(W[i,:])
    
    return Adj, W                                                   # Return adjacency and weight matrices


# Function to compute quadratic cost function and its gradient for distributed consensus problems
def cost_fcn_consensus(zz, QQ, rr):
    """
    Compute quadratic cost function and its gradient for distributed consensus problems.
    
    Implements the standard quadratic cost: f(z) = 0.5 * z^T * Q * z + r^T * z
    This is commonly used in distributed optimization where each agent has a local
    quadratic objective function.
    
    Args:
        zz (np.array): Current state vector. Shape: (d,)
        QQ (np.array): Positive definite weight matrix. Shape: (d, d)
        rr (np.array): Linear coefficient vector. Shape: (d,)
        
    Returns:
        tuple: (cost_i, grad) where:
               - cost_i: Scalar cost function value
               - grad: Gradient vector (d-dimensional). Shape: (d,)
    """
    cost_i = 0.5 * zz.T @ QQ @ zz + rr.T @ zz                      # Quadratic cost: 0.5*z^T*Q*z + r^T*z
    grad = QQ @ zz + rr                                             # Gradient: Q*z + r
    return cost_i, grad                                             # Return cost value and gradient vector


# Function to compute local cost function for target localization based on distance measurements
def cost_fcn_local(z, p_i, d_i):
    """
    Compute local cost function for target localization based on distance measurements.
    
    Each agent measures distances to targets and tries to estimate target positions.
    The cost penalizes deviations between predicted and measured squared distances.
    Cost: sum over targets of (d_measured^2 - ||z_target - p_agent||^2)^2
    
    Args:
        z (np.array): Estimated target positions. Shape: (N_T, d)
        p_i (np.array): Agent i's position. Shape: (d,)
        d_i (np.array): Agent i's distance measurements to all targets. Shape: (N_T,)
        
    Returns:
        tuple: (cost_i, grad) where:
               - cost_i: Scalar local cost for agent i
               - grad: Gradient w.r.t. target positions. Shape: (N_T, d)
    """
    cost_i = 0                                                      # Initialize local cost accumulator
    grad = np.zeros_like(z)                                         # Initialize gradient with same shape as z

    for tau in range(N_T):                                          # For each target tau
        diff = z[tau] - p_i                                         # Vector from agent to estimated target position
        norm_sq = np.dot(diff.T, diff)                              # Squared distance ||z_tau - p_i||^2
        err = d_i[tau]**2 - norm_sq                                 # Error: measured_distance^2 - predicted_distance^2
        cost_i += err**2                                            # Accumulate squared error cost
        grad[tau] += -4 * err * diff                                # Gradient: -4 * error * (z_tau - p_i)
    
    return cost_i, grad                                             # Return total cost and gradient


# Algorithm for distributed consensus using gradient tracking
def distributed_consensus(graph, name, Q, r, cost_opt, z_opt):
    """
    Implement distributed consensus algorithm using gradient tracking.
    
    Solves distributed optimization problem where agents collaborate to minimize
    sum of local quadratic functions while communicating only with neighbors.
    Uses gradient tracking to handle time-varying graphs and achieve exact convergence.
    
    Args:
        graph (networkx.Graph): Communication topology between agents
        name (str): Graph name for plotting and identification
        Q (np.array): List of local quadratic matrices Q_i for each agent. Shape: (NN, d, d)
        r (np.array): List of local linear vectors r_i for each agent. Shape: (NN, d)
        cost_opt (float): Optimal cost value for convergence analysis
        z_opt (np.array): Optimal state vector for convergence analysis. Shape: (d,)
        
    Returns:
        cost (np.array): Cost history over iterations. Shape: (max_iters_1_1-1,)
        z_avg (np.array): Average state of agents at final iteration. Shape: (d,)
    """
    Adj, A = get_matrices(graph, NN)                               # Get adjacency and Metropolis-Hastings weight matrices
    
    cost = np.zeros(max_iters_1_1 - 1)                             # Cost history over iterations
    z = np.zeros((max_iters_1_1, NN, d))                           # State variables: z[k, i, :] = state of agent i at iteration k
    s = np.zeros((max_iters_1_1, NN, d))                           # Gradient tracking variables for EXTRA algorithm
    grad_new = np.zeros((max_iters_1_1 - 1, NN, d))                # Storage for new gradients at each iteration

    # Initialize gradient tracking variables with initial gradients
    for i in range(NN):                                            # For each agent i
        _, s[0, i] = cost_fcn_consensus(z[0, i], Q[i], r[i])       # s[0,i] = gradient of f_i at z[0,i]

    for k in range(max_iters_1_1 - 1):                             # Main iteration loop
        # --- State update step: z[k+1] = consensus(z[k]) - alpha * s[k] ---
        for i in range(NN):                                        # For each agent i
            z[k + 1, i] = A[i, i] * z[k, i]                        # Start with weighted self-state
            N_i = np.nonzero(Adj[i])[0]                            # Get indices of neighbors of agent i
            for j in N_i:                                          # For each neighbor j of agent i
                z[k + 1, i] += A[i, j] * z[k, j]                   # Add weighted neighbor state
            z[k + 1, i] -= alpha_1_1 * s[k, i]                     # Subtract gradient step

        # --- Gradient tracking update: s[k+1] = consensus(s[k]) + grad_new - grad_old ---
        for i in range(NN):                                        # For each agent i
            s[k + 1, i] = A[i, i] * s[k, i]                        # Start with weighted self-gradient-tracking
            N_i = np.nonzero(Adj[i])[0]                            # Get indices of neighbors of agent i
            for j in N_i:                                          # For each neighbor j of agent i
                s[k + 1, i] += A[i, j] * s[k, j]                   # Add weighted neighbor gradient-tracking

            _, grad_new[k, i] = cost_fcn_consensus(z[k + 1, i], Q[i], r[i])  # Compute new gradient at updated state
            _, grad_old = cost_fcn_consensus(z[k, i], Q[i], r[i])            # Compute old gradient at previous state
            s[k + 1, i] += grad_new[k, i] - grad_old                         # Add gradient difference for tracking

            ell_i, _ = cost_fcn_consensus(z[k, i], Q[i], r[i])               # Compute local cost for monitoring
            cost[k] += ell_i                                                 # Accumulate total cost across all agents
    # --- Compute average state for convergence analysis ---
    z_avg = np.mean(z[-1, :, :], axis=0)                           # Average state across all agents at each iteration
    # Compute convergence metrics for analysis
    grad_tot = np.sum(grad_new, axis=1)                            # Sum gradients across all agents at each iteration
    grad_norm = np.linalg.norm(grad_tot, axis=1)                   # Compute norm of summed gradients (optimality measure)

    plot.plot_results_distributed(cost-cost_opt, grad_norm, z, z_opt, name + " graph")  # Plot convergence results

    return cost, z_avg                                             # Return cost history for analysis and final average state


# Algorithm for centralized target localization using gradient descent
def centralized_target_localization(distances, agent_positions, z_true):
    """
    Solve target localization problem using centralized gradient descent.
    
    All agents share their distance measurements to jointly estimate target positions.
    This serves as a performance benchmark for the distributed version.
    Minimizes sum of squared errors between measured and predicted distances.
    
    Args:
        distances (np.array): Distance measurements from each agent to each target. Shape: (NN, N_T)
        agent_positions (np.array): Known positions of all agents. Shape: (NN, d)
        z_true (np.array): True target positions for comparison. Shape: (N_T, d)
        
    Returns:
        tuple: (z_final, cost_tot) where:
               - z_final: Final estimated target positions. Shape: (N_T, d)
               - cost_tot: Total cost history over iterations
    """
    z = np.zeros((max_iters_1_2, N_T, d))                           # Target position estimates over time
    z[0] = np.ones((N_T, d)) * 5                                      # Initialize with average agent position
    grad_acc = np.zeros((max_iters_1_2-1, N_T, d))                  # Accumulated gradients from all agents
    cost = np.zeros((max_iters_1_2-1, NN))                          # Individual agent costs over time

    for k in range(max_iters_1_2 - 1):                              # Main optimization loop
        grad_acc[k] = 0                                             # Reset gradient accumulator for this iteration
        
        for i in range(NN):                                         # For each agent i
            _, grad = cost_fcn_local(z[k], agent_positions[i], distances[i])  # Compute agent i's gradient
            grad_acc[k] += grad                                     # Accumulate gradient from agent i
            cost[k, i], _ = cost_fcn_local(z[k], agent_positions[i], distances[i])  # Compute agent i's local cost for monitoring

            
        z[k + 1] = z[k] - alpha_1_2 * grad_acc[k]                   # Update target estimates using accumulated gradient


    cost_tot = np.sum(cost, axis=1)                                 # Total cost across all agents

    plot.plot_results_centralized(cost_tot, grad_acc)              # Plot cost and gradient convergence
    plot.plot_results_localization(z[-1, :, :], agent_positions, z_true, 'Centralized ')  # Plot final localization result

    return z[-1, :, :] , cost_tot                                   # Return final estimates and cost history


# Algorithm for distributed target localization using gradient tracking
def distributed_target_localization(graph, distances, agent_positions, z_true):
    """
    Solve target localization problem using distributed gradient tracking algorithm.
    
    Each agent maintains its own estimate of target positions and communicates only
    with neighbors. Uses gradient tracking (EXTRA) to achieve consensus on target locations
    while minimizing sum of local distance-based cost functions.
    
    Args:
        graph (networkx.Graph): Communication topology between agents
        distances (np.array): Distance measurements from each agent to targets. Shape: (NN, N_T)
        agent_positions (np.array): Known positions of all agents. Shape: (NN, d)
        z_true (np.array): True target positions for performance evaluation. Shape: (N_T, d)
        
    Returns:
        tuple: (z_estimates, cost) where:
               - z_estimates: Final average target position estimates. Shape: (N_T, d)
               - cost: Local cost history for each agent. Shape: (max_iters_1_2-1, NN)
    """
    NN = graph.number_of_nodes()                                    # Number of agents in the network
    N_T = distances.shape[1]                                        # Number of targets to localize
    d = agent_positions.shape[1]                                    # Spatial dimension (2D or 3D)

    Adj, A = get_matrices(graph, NN)                                # Get adjacency and weight matrices for consensus

    z = np.zeros((max_iters_1_2, NN , N_T, d))                      # Target estimates: z[k,i,tau,:] = agent i's estimate of target tau at time k
    s = np.zeros((max_iters_1_2, NN, N_T, d))                       # Gradient tracking variables for distributed optimization
    cost_tot = np.zeros((max_iters_1_2 - 1))                          # Local cost for each agent over time
    grad_norm = np.zeros(max_iters_1_2 - 1)                         # Gradient norm for convergence monitoring
    z[0] = np.ones((NN, N_T, d)) * 5                                  # Initialize target estimates to average agent position   
    for i in range(NN):                                             # For each agent i
        _, s[0, i] = cost_fcn_local(z[0,i], agent_positions[i], distances[i])  # s[0,i] = initial gradient

    for k in range(max_iters_1_2 - 1):                              # Main distributed optimization loop
        grad_sum = 0                                                # Reset gradient sum for this iteration
        cost = 0                                                    # Reset total cost for this iteration

        # --- State update: z[k+1,i] = consensus(z[k]) - alpha * s[k,i] ---
        for i in range(NN):                                         # For each agent i
            z_sum = A[i, i] * z[k, i]                               # Start with weighted self-estimate
            for j in range(NN):                                     # For each potential neighbor j
                if Adj[i, j] == 1:                                  # If agent j is a neighbor of agent i
                    z_sum += A[i, j] * z[k, j]                      # Add weighted neighbor estimate
            z[k + 1, i] = z_sum - alpha_1_2 * s[k, i]               # Apply gradient step

        # --- Gradient tracking update: s[k+1,i] = consensus(s[k]) + grad_new - grad_old ---
        for i in range(NN):                                         # For each agent i
            s_sum = A[i, i] * s[k, i]                               # Start with weighted self-gradient-tracking
            for j in range(NN):                                     # For each potential neighbor j
                if Adj[i, j] == 1:                                  # If agent j is a neighbor of agent i
                    s_sum += A[i, j] * s[k, j]                      # Add weighted neighbor gradient-tracking
            
            _, grad_new = cost_fcn_local(z[k + 1, i], agent_positions[i], distances[i])  # New gradient at updated estimate
            _, grad_old = cost_fcn_local(z[k, i], agent_positions[i], distances[i])            # Old gradient at previous estimate
            s[k + 1, i] = s_sum + grad_new - grad_old                                     # Update gradient tracking

            grad_sum += grad_old                                    # Sum gradients for convergence analysis
            cost += cost_fcn_local(z[k, i], agent_positions[i], distances[i])[0]    # Compute local cost for monitoring
        
        grad_norm[k] = np.linalg.norm(grad_sum)                     # Norm of summed gradients (optimality measure)

        cost_tot[k] = cost                                          # Store total cost for this iteration

    z_avg = np.mean(z, axis=1)                                      # Average estimate across all agents
    z_estimates = z_avg[-1, :, :]                                   # Final consensus estimate

    
    plot.plot_results_distributed(cost_tot, grad_norm, z, z_opt=None, title="Distributed Target Localization")  # Plot convergence
    plot.plot_results_localization(z_estimates, agent_positions, z_true, 'Distributed')        # Plot final localization

    return z_estimates, cost_tot                                        # Return consensus estimates and cost history

# Algorithm for aggregative distributed optimization with collision avoidance
def aggregative(z_init, t_i, avoidance=True):
    """
    Implement aggregative distributed algorithm with collision avoidance and time-varying communication.
    
    Agents move towards individual target positions while maintaining formation consensus
    and avoiding collisions. Communication graph changes based on proximity, creating
    a challenging distributed optimization problem with space-time varying topology.
    
    The algorithm minimizes: sum_i [||z_i - t_i||^2 + gamma*||z_i - s_i||^2] + collision penalties
    where s_i is the local estimate of the aggregate (mean position).
    
    Args:
        z_init (np.array): Initial agent positions. Shape: (N, d)
        t_i (np.array): Target positions for each agent. Shape: (N, d)
        avoidance (bool): Enable collision avoidance between agents (default: True)
        
    Returns:
        tuple: (z_dist, cost_dist, grad_z, graphs) where:
               - z_dist: Position trajectory for all agents. Shape: (max_iters_2_1, N, d)
               - cost_dist: Total cost over iterations. Shape: (max_iters_2_1-1,)
               - grad_z: Gradient history for analysis. Shape: (max_iters_2_1-1, N, d)
               - graphs: Communication graphs at each iteration (list of networkx.Graph)
    """
    N, d = z_init.shape                                             # Number of agents and spatial dimension
    z_dist = np.zeros((max_iters_2_1, N, d))                        # Agent positions over time
    s_dist = np.zeros((max_iters_2_1, N, d))                        # Local aggregate estimates (consensus variables)
    v_dist = np.zeros((max_iters_2_1, N, d))                        # Gradient tracking variables for aggregate term
    grad_z = np.zeros((max_iters_2_1 - 1, N, d))                    # Gradient history for analysis
    cost_dist = np.zeros(max_iters_2_1 - 1)                         # Total cost over iterations
    z_dist[0] = z_init                                              # Set initial positions
    s_dist[0] = z_init.copy()                                       # Initialize aggregate estimates to initial positions
    graphs = []                                                     # Store communication graphs for analysis
    alpha = alpha_2_1                                               # Step size (could be adaptive)

    # Initialize gradient tracking variables
    for i in range(N):                                              # For each agent i
        grad_sigma = -2 * gamma * (z_dist[0, i] - s_dist[0,i])      # Gradient of gamma*||z_i - s_i||^2 term
        v_dist[0, i] = grad_sigma                                   # Initialize tracking variable
    
    grad1 = np.zeros((max_iters_2_1 - 1, d))                    # Initialize gradient tracking for target term
    grad2 = np.zeros((max_iters_2_1 - 1, d))                    # Initialize gradient tracking for aggregative term
    
    for k in range(max_iters_2_1 - 1):                              # Main algorithm loop
        
        # --- Build time-varying communication graph based on proximity ---
        G_comm = nx.Graph()                                         # Create new communication graph
        G_comm.add_nodes_from(range(N))                             # Add all agents as nodes

        for i in range(N):                                          # For each agent i
            for j in range(i+1, N):                                 # For each other agent j > i
                dist = np.linalg.norm(z_dist[k, i] - z_dist[k, j])  # Distance between agents i and j
                if dist <= r_comm:                                  # If within communication range
                    G_comm.add_edge(i, j)                           # Add communication link

        graphs.append(G_comm)                                       # Store graph for analysis
        Adj, A = get_matrices(G_comm, N)                            # Get adjacency and weight matrices

        # --- Position update: z[k+1,i] = z[k,i] - alpha * (grad_target + grad_aggregative + grad_collision) ---
        for i in range(N):                                          # For each agent i
            # Compute gradient components
            grad = 2 * (z_dist[k, i] - t_i[i]) + 2 * gamma * (z_dist[k, i] - s_dist[k, i])  # Target + aggregative gradients
            
            if i == 0:
                grad1[k] = 2 * (z_dist[k, i] - t_i[i]) 
                grad2[k] = 2 * gamma * (z_dist[k, i] - s_dist[k, i]) 


            if avoidance == True:                                   # If collision avoidance is enabled
                for j in G_comm.neighbors(i):                       # For each neighbor j of agent i
                    if j == i:                                      # Skip self (shouldn't happen in neighbors)
                        continue
                    diff = z_dist[k, i] - z_dist[k, j]              # Vector from j to i
                    dist = np.linalg.norm(diff)                     # Distance between agents
                    hij = max(0, delta - dist)                          # Collision avoidance term (non-negative)
                    grad += (-4 * beta * hij / dist) * diff if dist > 0 else 0  # Add repulsive force (avoid division by zero) 

            update = grad + v_dist[k, i]                            # Total update includes gradient tracking

            z_dist[k + 1, i] = z_dist[k, i] - alpha * update        # Apply gradient step

        # --- Aggregate estimate update: s[k+1,i] = consensus(s[k]) + (z[k+1,i] - z[k,i]) ---
        for i in range(N):                                          # For each agent i
            s_new = A[i,i]*s_dist[k,i] + sum(A[i,j] * s_dist[k,j] for j in range(N) if Adj[i,j])  # Consensus step
            s_dist[k+1,i] = s_new + (z_dist[k+1,i] - z_dist[k,i])   # Add position change for tracking

        # --- Gradient tracking update: v[k+1,i] = consensus(v[k]) + (grad_new - grad_old) ---
        for i in range(N):                                          # For each agent i
            grad_new = -2*gamma*(z_dist[k+1,i] - s_dist[k+1,i])     # New gradient of consensus term
            grad_old = -2*gamma*(z_dist[k,i]   - s_dist[k,i])       # Old gradient of consensus term
            v_new = A[i,i]*v_dist[k,i] + sum(A[i,j]*v_dist[k,j] for j in range(N) if Adj[i,j])  # Consensus step
            v_dist[k+1,i] = v_new + (grad_new - grad_old)           # Add gradient difference for tracking

        # --- Compute total cost for monitoring ---
        cost = 0.0                                                  # Initialize total cost
        grad_zk = np.zeros((N, d))                                 # Initialize gradient for this iteration 
        sigma = np.mean(z_dist[k], axis=0)                # Compute local aggregate estimate (mean position)
        for i in range(N):                                          # For each agent i
            cost += np.linalg.norm(z_dist[k, i] - t_i[i]) ** 2      # Target tracking cost
            cost += gamma * np.linalg.norm(z_dist[k, i] - sigma) ** 2  # Aggregative cost
            grad_zk[i] = 2 * (z_dist[k, i] - t_i[i]) + 2 * gamma * (z_dist[k, i] - sigma)
            if avoidance == True:                                   # If collision avoidance enabled
                for j in G_comm.neighbors(i):                       # For each neighbor j of agent i
                    if j == i:                                      # Skip self
                        continue
                    diff = z_dist[k, i] - z_dist[k, j]              # Vector from j to i
                    dist = np.linalg.norm(diff)                     # Distance between agents
                    hij = max(0, delta - dist)                      # Collision avoidance term (non-negative)
                    cost += beta * hij ** 2
                    grad_zk[i] += (-4 * beta * hij / dist) * diff if dist > 0 else np.zeros_like(diff)  # Add repulsive force

        cost_dist[k] = cost                                         # Store total cost
        grad_z[k] = grad_zk

    return z_dist, cost_dist, grad_z, graphs                       # Return all results for analysis


