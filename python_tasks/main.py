# Main of the DAS project
# Authors: Group 23 - Niccolò Antolini, Daniele Crivellari, Brando Ulissi

# Import the necessary libraries and parameters
import numpy as np

from config import *
import functions as f
import plots as plot

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)



##########################################################
######################  CODE SETUP  ######################
##########################################################

# Flag to set to True or False to execute the different Project tasks
TASK1_1 = True                                                     # Distributed consensus optimization
TASK1_2 = True                                                      # Distributed target localization
TASK2_1 = True                                                      # Aggregative optimization


# Flag for configuring the execution
GRAPH_1_2 = "Cycle"                                                 # Graph topology for task 1.2 (distributed target localization)
                                                                    # Possible values: "Erdos-Renyi", "Star", "Ladder", "Path", "Cycle", "Complete"

AVOIDANCE_2_1 = True                                                    # Flag for collision avoidance in aggregative optimization


##########################################################
##############  PRELIMINARY: GRAPH SETUP  ###############
##########################################################

if TASK1_1 or TASK1_2:
    print(f"\033[1;36m\n\nPRELIMINARY: GRAPH GENERATION\033[0m")
    
    np.random.seed(42)                                              # Set seed for reproducible graph generation
    graphs = f.create_graphs()                                      # Generate graphs with NN nodes
    plot.plot_graph(graphs)                                         # Plot the generated graphs for visualization



##########################################################
##########  TASK 1.1: DISTRIBUTED CONSENSUS OPT  #########
##########################################################

if TASK1_1:                                                         # If the corresponding flag is activated, run Task 1.1
    
    print(f"\033[1;36m\n\nTASK 1.1: DISTRIBUTED CONSENSUS OPTIMIZATION\033[0m")
    
    # Set specific seed for reproducible results in Task 1.1
    np.random.seed(10)
    
    # Initialize quadratic and linear cost function parameters for each node
    Q = []                                                          # List to store quadratic coefficient matrices
    r = []                                                          # List to store linear coefficient vectors
    
    for i in range(NN):
        [Q11, Q22] = np.random.uniform(0, 1, 2)                     # Random quadratic coefficients for node i
        [r1, r2] = np.random.normal(0, 1, 2)                       # Random linear coefficients for node i
        Q.append(np.diag([Q11, Q22]))                               # Create 2x2 diagonal matrix for node i
        r.append(np.array([r1, r2]))                                # Create linear coefficient vector for node i
    Q = np.array(Q)
    r = np.array(r)

    # Compute centralized optimal solution for comparison
    Qcentr = sum(Q)                                                # Sum of all quadratic matrices
    rcentr = sum(r)                                                # Sum of all linear vectors
    z_opt = - np.linalg.inv(Qcentr) @ rcentr                       # Centralized optimal solution
    cost_opt, _ = f.cost_fcn_consensus(z_opt, Qcentr, rcentr)      # Centralized optimal cost

    print(f"\n{'='*40}\nCentralized Optimal Solution\n{'='*40}")
    print(f"z_opt:\n{z_opt}\n")
    print(f"Optimal cost: {cost_opt:.4f}\n{'='*40}\n")

    # Initialize storage for distributed results
    costs = []                                                      # List to store costs for each graph topology
    z_avgs = []                                                      # List to store average solutions for each graph topology
    # Run distributed gradient tracking for each graph topology
    for graph in graphs:
        cost, z_avg = f.distributed_consensus(
            graph,
            graph_names[graphs.index(graph)],
            Q, 
            r,
            cost_opt,
            z_opt
        )
        costs.append(cost)                                          # Store cost evolution for current graph
        z_avgs.append(z_avg)                                         # Store average solution for current graph

    # Plot comparison of consensus results across different graph topologies
    plot.plot_results_consensus(costs, cost_opt, z_avgs, z_opt)



##########################################################
#######  TASK 1.2: DISTRIBUTED TARGET LOCALIZATION  #####
##########################################################

if TASK1_2:                                                         # If the corresponding flag is activated, run Task 1.2
    
    print(f"\033[1;36m\n\nTASK 1.2: DISTRIBUTED TARGET LOCALIZATION\033[0m")
    
    # Set specific seed for reproducible results in Task 1.2
    np.random.seed(20)

    # ====== INITIAL SETUP ======
    p_i = np.random.uniform(0, 10, size=(NN, d))                   # Robot positions (NN robots in d dimensions)
    z_true = np.random.uniform(0, 10, size=(N_T, d))               # True target positions (N_T targets in d dimensions)

    # ====== GENERATE NOISY DISTANCE MEASUREMENTS ======
    d_i = np.zeros((NN, N_T))                                      # Matrix to store noisy distance measurements
    for i in range(NN):
        for tau in range(N_T):
            true_dist = np.linalg.norm(z_true[tau] - p_i[i])        # Compute true Euclidean distance
            d_i[i, tau] = true_dist + np.random.normal(0, noise_std) # Add Gaussian noise to measurement

    # Select graph topology for distributed algorithm
    g = graphs[graph_names.index(GRAPH_1_2)]                          # Use Cycle graph topology for this task

    # ====== CENTRALIZED TARGET LOCALIZATION (BASELINE) ======
    z_est_centralized, cost_centralized = f.centralized_target_localization( 
        d_i,                                                        # Noisy distance measurements
        p_i,                                                        # Robot positions
        z_true,                                                     # True target positions (for comparison)
    )

    # ====== DISTRIBUTED TARGET LOCALIZATION ======
    z_avg_distributed, cost_distributed = f.distributed_target_localization(
        g,                                                          # Graph topology
        d_i,                                                        # Noisy distance measurements
        p_i,                                                        # Robot positions
        z_true,                                                     # True target positions (for comparison)
    )

    # Plot comparison between centralized and distributed approaches
    plot.plot_comparisons(
        cost_centralized,                                           # Centralized cost evolution
        cost_distributed,                                           # Distributed cost evolution
        z_est_centralized,                                          # Centralized target estimates
        z_avg_distributed,                                          # Distributed target estimates
        z_true,                                                     # True target positions
        p_i,                                                        # Robot positions
    )



##########################################################
##########  TASK 2.1: AGGREGATIVE OPTIMIZATION  ##########
##########################################################

if TASK2_1:                                                         # If the corresponding flag is activated, run Task 2.1
    
    print(f"\033[1;36m\n\nTASK 2.1: AGGREGATIVE OPTIMIZATION\033[0m")
    
    # Set specific seed for reproducible results in Task 2.1
    np.random.seed(3)

    # ====== INITIAL SETUP ======
    z_init = np.random.uniform(0, 10, size=(N, d))                 # Starting positions of robots
    t_i = np.random.uniform(0, 10, size=(N, d))                    # Personal target positions for each robot

    # ====== RUN AGGREGATIVE OPTIMIZATION ======
    z_dist, cost_dist, grad_dist, graphs = f.aggregative(
        z_init,                                                     # Initial robot positions
        t_i,                                                        # Target positions
        avoidance=AVOIDANCE_2_1,                                        # Enable/disable collision avoidance
    )

    # ====== VISUALIZATION AND ANALYSIS ======
    plot.plot_aggregative(cost_dist, grad_dist, z_dist, t_i)       # Plot optimization results (cost, gradient, trajectories)
    plot.animation(z_dist[::1], t_i, graphs[::1])                # Animate robot trajectories (subsampled for performance)

print(f"\033[1;32m\nALL TASKS COMPLETED SUCCESSFULLY!\033[0m\n\n")