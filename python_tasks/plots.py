# Plotting functions for the DAS project
# Authors: Group 23 - Niccolò Antolini, Daniele Crivellari, Brando Ulissi

# Import the necessary libraries and parameters
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as patches

from config import *  


# Plot for different graph topologies used
def plot_graph(graphs):
    """
    Visualize multiple communication graph topologies in a grid layout.

    Displays different network topologies used in distributed algorithms to allow
    visual comparison of their connectivity patterns.

    Args:
        graphs (list of networkx.Graph): List of Graphs representing different topologies. Shape: (6,).
    """

    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))         # Create a 2x3 grid of subplots
    fig.suptitle("Graph Topologies", fontsize=25, fontweight='bold')  # Set the main title for the entire figure
    axes = axes.flatten()                                     # Flatten 2D axes array for easy iteration

    # Plot each graph in its corresponding subplot
    for i, (graph, name) in enumerate(zip(graphs, graph_names)):
        nx.draw(graph, with_labels=True, ax=axes[i])          # Draw graph with node labels on subplot
        axes[i].set_title(name, fontsize=20)                               # Set individual subplot title

    plt.tight_layout()                                        # Adjust spacing to prevent overlap
    plt.show()                                                # Display the complete figure


# Plot convergence results for distributed consensus algorithms
def plot_results_consensus(costs, cost_opt, z_avgs, z_opt):
    """
    Analyze convergence behavior of consensus algorithms on various graph topologies.

    Plots optimality errors over iterations and compares the final performance
    of each topology in terms of cost and estimate accuracy.

    Args:
        costs (list): List of cost trajectories, one per graph topology. Shape: (6, max_iters_1_1,).
        cost_opt (list): Optimal cost value used for computing the error. Shape: (max_iters_1_1,).
        z_avgs (list): Final average estimates for each topology. Shape: (6, d,).
        z_opt (np.array): Ground truth optimal estimate. Shape: (d,).
    """
    
    # Convergence comparison plot
    plt.figure(figsize=(12, 6))                               # Create a new figure with specified size
    plt.title("Convergence Comparison Across Different Graph Topologies", fontsize=16, fontweight='bold')  # Main title
    plt.xlabel(r"$k$", fontsize=14)                    # X-axis label for iteration number
    plt.ylabel(r"$\ell(z) - \ell(z^\star)$", fontsize=14)               # Y-axis label for error magnitude
    plt.grid(True)                                            # Enable grid for better readability

    # Plot optimality error for each topology on log scale
    for i, (cost, name) in enumerate(zip(costs, graph_names)):
        plt.semilogy(np.abs(cost - cost_opt), label=name)     # Log-scale plot of |cost - optimal_cost|

    plt.legend()                                              # Display legend with topology names
    plt.show()                                                # Display the plot

    # Print numerical summary of final convergence performance
    print("\nFinal optimality errors for each topology:")      # Header for numerical results
    print("----------------------------------------")
    for name, cost, z_avg in zip(graph_names, costs, z_avgs):
        final_error = np.abs(cost[-1] - cost_opt)             # Calculate final iteration error
        estimate_error = np.linalg.norm(z_avg - z_opt)        # Calculate norm between average estimate and optimal estimate
        print(f"{name:12s}: Cost error = {final_error:.2e}, Estimate error = {estimate_error:.2e}")  # Print both errors


# Plot convergence results for centralized target localization algorithms
def plot_results_centralized(cost_tot, grad_acc):
    """
    Evaluate convergence metrics for centralized target localization.

    Provides a visual analysis of cost evolution and gradient norm decay over iterations
    for centralized optimization.

    Args:
        cost_tot (np.array): Total cost at each iteration. Shape: (max_iters_1_2,)._
        grad_acc (np.array): Gradient tensors. Shape: (max_iters_1_2, N_T, d,).
    """

    grad_norm = np.linalg.norm(grad_acc, axis=(1, 2))         # Compute L2 norm of gradients across spatial dimensions

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))           # Create side-by-side subplot layout
    fig.suptitle("Convergence of Centralized Target Localization", fontsize=16, fontweight='bold')  # Main figure title

    # Left panel: Total cost evolution
    axes[0].semilogy(np.abs(cost_tot), label="Total cost")    # Log-scale plot of absolute cost values
    axes[0].set_xlabel(r"$k$")                          # X-axis label
    axes[0].set_ylabel(r"$\ell(z)$")                          # Y-axis label
    axes[0].grid()                                            # Enable grid

    # Right panel: Gradient norm evolution
    axes[1].semilogy(grad_norm, label="Gradient norm")        # Log-scale plot of gradient magnitude
    axes[1].set_xlabel(r"$k$")                                   # X-axis label
    axes[1].set_ylabel(r"$\|| \nabla \ell(z) \|| $")                          # Y-axis label
    axes[1].grid()                                            # Enable grid

    plt.tight_layout()                                        # Optimize subplot spacing
    plt.show()                                                # Display the figure


# Plot convergence results for distributed target localization algorithms
def plot_results_distributed(cost, grad_norm, z, z_opt, title):
    """
    Evaluate convergence behavior of distributed localization algorithms.

    Produces a 2x2 panel analysis showing total cost, gradient norm decay,
    consensus error between agents, and evolution of estimates over time.

    Args:
        cost (np.array): Total cost values over iterations. Shape: (max_iters,).
        grad_norm (np.array): L2 norm of gradients over iterations. Shape: (max_iters,).
        z (np.array): Agent estimates over time. Shape: (max_iters, NN, [N_T,] d).
        z_opt (np.array): Ground truth optimal target location (optional). Shape: ([N_T,] d).
        title (str): Algorithm name to annotate the plot.
    """
    
    NN = z.shape[1]                                           # Number of agents/agents
    if len(z.shape) > 3:
        N_T = z.shape[2]                                      # Number of targets (for multi-target scenarios)
    d = z.shape[-1]                                           # Dimensionality of the estimation space

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))          # Create 2x2 grid of subplots
    fig.suptitle("Convergence of " + title, fontsize=16, fontweight='bold')  # Main title with algorithm name
    axes = axes.flatten()                                     # Flatten for easy indexing

    # Panel 1: Total cost evolution
    axes[0].semilogy(np.abs(cost), label="Total cost")        # Log-scale plot of absolute cost
    axes[0].set_xlabel(r"$k$")                                   # X-axis label
    if len(z.shape) == 4:
        axes[0].set_ylabel(r"$\ell(z)$")         # Y-axis label in mathematical notation
    else:
        axes[0].set_ylabel(r"$\ell(z)-\ell(z^*)$")         # Y-axis label in mathematical notation
    axes[0].grid(True)                                        # Enable grid

    # Panel 2: Gradient norm evolution
    axes[1].semilogy(grad_norm, label="Gradient norm")        # Log-scale plot of gradient magnitude
    axes[1].set_xlabel(r"$k$")                                   # X-axis label
    axes[1].set_ylabel(r"$\||\nabla \ell(z)\||$")                     # Y-axis label (mathematical notation)
    axes[1].grid(True)                                        # Enable grid

    # Panel 3: Consensus error analysis
    z_avg = np.mean(z, axis=(1))                              # Compute average estimate across all agents

    # Plot consensus error for each agent
    for i in range(NN):
        if len(z.shape) == 4:
            # For multi-target case, compute norm across targets and dimensions
            diff = np.linalg.norm(z[1:, i] - z_avg[1:], axis=(1, 2))  
        else:
            # For single target case, compute norm across dimensions only
            diff = np.linalg.norm(z[1:, i] - z_avg[1:, :], axis=(1))  
            
        axes[2].semilogy(diff, label=f"Agent {i}")            # Log-scale plot of consensus error
    axes[2].legend()                                          # Show legend for each agent
    axes[2].set_xlabel(r"$k$")                                   # X-axis label
    axes[2].set_ylabel(r"$z_i-z_{\mathrm{avg}}$")               # Y-axis label
    axes[2].grid(True)                                        # Enable grid

    # Panel 4: Individual agent estimates over time
    # Define custom color palette for better visualization
    custom_colors = [
        '#a8ddb5',  # light mint green
        '#006d2c',  # dark green
        '#9ecae1',  # light blue
        '#08519c',  # dark blue
        '#fdae6b',  # light orange
        '#e6550d'   # dark orange
    ]

    # Plot z_opt for each dimension
    if z_opt is not None:
        for k in range(d):
            color = custom_colors[0] if k % 2 == 0 else custom_colors[2]
            axes[3].hlines(z_opt[k], 0, z.shape[0]-1, colors=color, linewidth=2,
                           label=fr"$z^*_{{{k+1}}}$")

    # Plot estimates for each agent
    dict = {0 : 'x', 1 : 'y'}
    for i in range(NN):
        if len(z.shape) == 4:
            # Multi-target case: plot each target and dimension separately
            for j in range(N_T):
                for k in range(d):
                    z_plot = np.concatenate(([z[0, i, j, k]], z[:, i, j, k]))
                    color_idx = (j * d + k) % len(custom_colors)
                    # Only add label for first agent to avoid legend clutter
                    label = fr"$z_{{{j+1},{dict[k]}}}$" if i == 0 else None
                    axes[3].plot(z_plot, color=custom_colors[color_idx], 
                               linestyle='--', label=label)
    # Move legend to the right outside the plot
                    axes[3].legend(loc='center right', fontsize=11)
        else:
            # Single target case: plot each dimension separately
            for k in range(d):
                z_plot = np.concatenate(([z[0, i, k]], z[:, i, k]))
                color_idx = k * 2 + 1
                # Only add label for first agent to avoid legend clutter
                label = fr"$z_{{i,{k+1}}}$" if i == 0 else None
                axes[3].plot(z_plot, color=custom_colors[color_idx], 
                           linestyle=':', label=label)
        
    axes[3].legend()                                          # Show legend
    axes[3].set_xlabel(r"$k$")                                   # X-axis label
    axes[3].set_ylabel(r"$z_i$")                 # Y-axis label with subscript
    axes[3].grid(True)                                        # Enable grid
    axes[3].set_xscale('log')                                 # Set x-axis to logarithmic scale

    fig.subplots_adjust()               # Optimize layout spacing
    plt.show()                                                # Display the complete figure


# Plot final results of target localization
def plot_results_localization(z_est, agent_positions, z_true, method):
    """
    Visualize final estimated vs. true target positions.

    Compares estimated and actual positions of targets in a 2D space, along with
    agent locations, to assess localization accuracy.

    Args:
        z_est (np.array): Estimated target positions. Shape: (N_T, d).
        agent_positions (np.array): Positions of agents. Shape: (NN, d).
        z_true (np.array): True positions of targets. Shape: (N_T, d).
        method (str): Label describing the localization method used.
    """

    N_T = z_est.shape[0]                                      # Number of targets to localize
    plt.figure(figsize=(8, 8))                                 # Create square figure for spatial plot
    
    # True target positions (black circles)
    plt.scatter(z_true[:, 0], z_true[:, 1], c='black', marker='o', s=50, 
            label="True target")  # Label only first target for legend
    # Estimated target positions (red markers)
    plt.scatter(z_est[:, 0], z_est[:, 1], c='red', marker='x', s=100, 
                label="Final estimate") # Label only first estimate for legend
    
    # Plot agent positions (green square dots)
    plt.scatter(agent_positions[:, 0], agent_positions[:, 1], c='g', marker='s', s=100, label='Agents')
    
    plt.legend(fontsize=12, loc='upper left')                # Show legend in upper right
    plt.title(method + " Multi-Agent Target Localization", fontsize=16, fontweight='bold')  # Title with method name
    plt.xlabel("X Coordinate", fontsize=14)                   # X-axis label
    plt.ylabel("Y Coordinate", fontsize=14)                   # Y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)                 # Enable grid with subtle styling
    plt.axis('equal')                                         # Ensure equal aspect ratio for spatial accuracy
    plt.xlim(-0.1, 10.1)                                      # Set X-axis limits
    plt.ylim(-0.1, 10.1)                                      # Set Y-axis limits
    plt.show()                                                # Display the plot


# Plot comprehensive comparison between centralized and distributed localization approaches
def plot_comparisons(cost_centralized, cost_distributed, z_estimates_centralized, z_avg_distributed, z_true, agent_positions):
    """
    Compare centralized and distributed localization algorithms in terms of convergence and accuracy.

    This function visualizes cost convergence over iterations and spatial accuracy of the 
    estimated target positions from both centralized and distributed approaches.

    Args:
        cost_centralized (np.array): Cost history of the centralized algorithm. Shape: (max_iters_1_2,)
        cost_distributed (np.array): Cost history of the distributed algorithm. Shape: (max_iters_1_2,)
        z_estimates_centralized (np.array): Final centralized estimates of target positions. Shape: (N_T, d)
        z_avg_distributed (np.array): Final distributed (averaged) estimates of target positions. Shape: (N_T, d)
        z_true (np.array): Ground truth target positions. Shape: (N_T, d)
        agent_positions (np.array): Known agent positions. Shape: (NN, d)
    """

    # Cost convergence comparison plot
    plt.figure(figsize=(12, 6))                              # Create figure for cost comparison
    plt.title("Cost Convergence: Centralized vs Distributed", fontsize=16, fontweight='bold')
    plt.xlabel(r"$k$", fontsize=14)                   # X-axis label
    plt.ylabel(r"$\ell(z)$", fontsize=14)                          # Y-axis label
    plt.grid(True)                                           # Enable grid
    
    # Plot centralized cost
    plt.semilogy(cost_centralized, label="Centralized cost", linestyle='-', color='blue')
    
    # Plot total distributed cost (sum across agents)
    plt.semilogy(cost_distributed, label="Distributed cost", 
                linestyle='-', color='red')
    
    plt.legend()                                             # Show legend
    plt.show()                                               # Display plot

    # Print numerical comparison of final estimates
    print("\nComparison between true and estimated positions:")
    print("----------------------------------------")
    for i, (z_t, z_c, z_d) in enumerate(zip(z_true, z_estimates_centralized, z_avg_distributed)):
        print(f"Target {i+1}:")
        print(f"  True position:             {z_t}")
        print(f"  Centralized estimate:      {z_c}")
        print(f"  Distributed estimate:      {z_d}")
        print(" ")

    # Spatial visualization comparing all approaches
    plt.figure(figsize=(8, 8))                             # Create square figure for spatial comparison
    
    # Plot true target positions
    plt.scatter(z_true[:, 0], z_true[:, 1], c='black', marker='o', s=50,
                label="True Targets")
    
    # Plot centralized estimates
    plt.scatter(z_estimates_centralized[:, 0], z_estimates_centralized[:, 1], 
               c='red', label="Centralized Estimates", marker='+', s=100)
    
    # Plot distributed estimates
    plt.scatter(z_avg_distributed[:, 0], z_avg_distributed[:, 1], 
               c='orange', label="Distributed Estimates", marker='x', s=100)
    
    # Plot agent positions for context
    plt.scatter(agent_positions[:, 0], agent_positions[:, 1], c='g', marker='s', s=100, label='Agents')
    
    plt.legend(fontsize=12, loc='upper left')               # Show legend
    plt.title("Cooperative Multi-Agent Target Localization", fontsize=16, fontweight='bold')
    plt.xlabel("X Coordinate", fontsize=14)                  # X-axis label
    plt.ylabel("Y Coordinate", fontsize=14)                  # Y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)                # Enable grid with subtle styling
    plt.axis('equal')                                        # Equal aspect ratio for spatial accuracy
    plt.show()                                               # Display the plot


# Plot results for aggregative optimization problems in multi-agent systems
def plot_aggregative(cost_dist, grad_dist, z_dist, t_i):
    """
    Analyze convergence and trajectory behavior in aggregative optimization problems.

    This function visualizes the convergence of cost and gradient norms during optimization
    and plots the individual agent trajectories towards their assigned targets.

    Args:
        cost_dist (np.array): Cost values over iterations. Shape: (max_iters_2_1,).
        grad_dist (np.array): Gradient vectors per agent per iteration. Shape: (max_iters_2_1, N, d,).
        z_dist (np.array): Agent positions over time. Shape: (max_iters_2_1, N, d,).
        t_i (np.array): Assigned target positions for each agent. Shape: (N, d).
    """

    # Convergence analysis plots
    fig, axes = plt.subplots(1,2, figsize=(12,5))           # Create side-by-side subplots
    fig.suptitle("Convergence of Aggregative Optimization", fontsize=16, fontweight='bold')
    
    # Left panel: Cost evolution
    axes[0].semilogy(np.abs(cost_dist), label="Total cost")        # Log-scale plot of absolute cost
    axes[0].set_ylabel(r"$\ell(z)$")                          # Y-axis label
    axes[0].set_xlabel(r"$k$")                                # X-axis label
    axes[0].grid(True)                                        # Enable grid
    
    # Right panel: Gradient norm evolution
    grad_norm = np.linalg.norm(grad_dist, axis=(1,2))         # Compute gradient norm across agents and dimensions
    axes[1].semilogy(grad_norm, label="Gradient norm")        # Log-scale plot of gradient magnitude
    axes[1].set_xlabel(r"$k$")                                # X-axis label
    axes[1].set_ylabel(r"$\||\nabla \ell(z)\||$")           # Y-axis label

    axes[1].grid(True)                                        # Enable grid

    plt.tight_layout()                                        # Optimize layout
    plt.show()                                                # Display convergence plots

    # Agent trajectory visualization
    plt.figure(figsize=(8, 8))                               # Create square figure for trajectory plot
    colors = plt.cm.get_cmap('tab10', N)                     # Get colormap with N distinct colors

    for i in range(N):
        # Plot trajectory of each agent
        plt.plot(z_dist[:, i, 0], z_dist[:, i, 1], label=f'Agent {i}', color=colors(i))  # Plot trajectory in agent-specific color
        # Plot target position for each agent
        plt.scatter(t_i[i, 0], t_i[i, 1], marker='x', color=colors(i), s=150, linewidths=2)
        
    plt.title("Agent Trajectories")                          # Plot title
    plt.xlabel("X")                                          # X-axis label
    plt.ylabel("Y")                                          # Y-axis label
    plt.axis('equal')                                        # Equal aspect ratio
    plt.xlim(-0.1, 10.1)                                     # Set axis limits
    plt.ylim(-0.1, 10.1)
    plt.grid(True, linestyle='--', alpha=0.7)                # Enable grid with subtle styling
    plt.legend()                                             # Show legend with agent labels
    plt.show()                                               # Display trajectory plot


# Animate the trajectories and communication links of agents in aggregative control
def animation(z_dist, targets, graph, dt=1):
    """
    Animate multi-agent system trajectories and communication during aggregative control.

    This function creates a frame-by-frame animation showing agent movements, communication 
    links, and convergence behavior as agents move toward their targets.

    Args:
        z_dist (np.array): Agent positions over time. Shape: (max_iters_2_1, N, d).
        targets (np.array): Target positions for each agent. Shape: (N, d).
        graph (list[networkx.Graph]): Sequence of communication topologies per time step. Shape: (max_iters_2_1,).
        dt (int, optional): Frame skip interval for animation speed (default is 1).
    """
    
    T = z_dist.shape[0]                      # Number of time steps
    XX = z_dist.reshape(T, N * 2)            # Reshape for easier access to x and y coordinates

    # Set up figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Aggregative Control animation", fontsize=16, fontweight='bold')
    ax.set_xlim(-0.1, 10.1)
    ax.set_ylim(-0.1, 10.1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Generate distinct colors for each agent
    colors = plt.cm.get_cmap('tab10', N)

    # Animate over time steps with interval dt
    for tt in range(0, T - dt, dt):
        ax.clear()                          # Clear previous frame content

        # Reset axis properties for the current frame
        ax.set_xlim(-0.1, 10.1)
        ax.set_ylim(-0.1, 10.1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        current_positions = []             # Store current positions of all agents

        # Plot agent trajectories and current positions
        for i in range(N):
            xs = XX[:tt + 1, 2 * i]        # X trajectory for agent i
            ys = XX[:tt + 1, 2 * i + 1]    # Y trajectory for agent i

            color = colors(i)

            # Plot trajectory with final marker in a single call
            ax.plot(xs, ys, lw=1, alpha=0.6, color=color,
                    marker='o', markevery=[-1], markersize=8,
                    markeredgecolor='k', markerfacecolor=color,
                    label=f'Agent {i + 1}')
            
            xi, yi = xs[-1], ys[-1]        # Current agent position
            current_positions.append((xi, yi))

            # Draw communication radius as a dotted circle
            comm_circle = patches.Circle((xi, yi), r_comm,
                                         fill=False, edgecolor=color,
                                         linestyle=':', linewidth=1.5, alpha=0.7)
            ax.add_patch(comm_circle)

        # Draw communication links between neighboring agents
        G_tt = graph[tt]
        for i, j in G_tt.edges():
            xi, yi = current_positions[i]
            xj, yj = current_positions[j]
            ax.plot([xi, xj], [yi, yj], 'k--', linewidth=1, alpha=0.5, 
                    label='Communications' if i == 0 and j == list(G_tt.neighbors(i))[0] else "")

        # Plot static target positions with color matching each agent
        for i in range(N):
            color = colors(i)
            ax.scatter(targets[i, 0], targets[i, 1], marker='x', s=150, 
                       color=color, linewidths=2)

        # Display current time step
        ax.set_title(fr"$k: {tt + 1} / {T}$")
        ax.legend(loc='upper left')        # Show legend with agent labels and communication tag
        plt.pause(0.1)                     # Pause for visualization timing
