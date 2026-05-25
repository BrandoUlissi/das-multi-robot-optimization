# Parameters for the the algorithms
# Authors: Group 23 - Niccolò Antolini, Daniele Crivellari, Brando Ulissi

# Import the necessary libraries and parameters
import numpy as np


###########################################
###########  SYSTEM PARAMETERS  ###########
###########################################

NN = 8                             # Number of nodes used in tasks 1.1 and 1.2
graph_names = ["Erdos-Renyi", "Star", "Ladder", "Path", "Cycle", "Complete"]  # Available graph topologies
graph_seed = 42                   # Seed for reproducibility in graph generation

d = 2                              # Dimension of the space (2D scenario)
N_T = 3                            # Number of target positions
noise_std = 0.05                   # Standard deviation of the additive measurement noise

N = 6                              # Number of robots for task 2.1



###########################################
#########  ALGORITHMS PARAMETERS  #########
###########################################

# TASK 1.1 PARAMETERS
max_iters_1_1 = 3000               # Maximum number of iterations for task 1.1
alpha_1_1 = 0.05                   # Step size for gradient updates in task 1.1

# TASK 1.2 PARAMETERS
max_iters_1_2 = 2000               # Maximum number of iterations for task 1.2
alpha_1_2 = 0.0005                 # Step size for gradient updates in task 1.2

# TASK 2.1 PARAMETERS
max_iters_2_1 = 1000              # Maximum number of iterations for task 2.1
alpha_2_1 = 0.02                 # Step size for optimization in task 2.1
gamma = 0.01                       # Gain parameter used in control or optimization updates
delta = 1                          # Minimum safety (security) distance between robots
beta = 2                           # Trade-off parameter for collision avoidance term
r_comm = 3.5                         # Maximum communication range between agents
