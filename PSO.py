# Author: Matt Kuehr / mck0063@auburn.edu

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.optimize import minimize

'''
Rastrigin Function (non-convex) BELOW
'''

# Define the bivariate Rastrigin function
def rastrigin_function(x):
    """
    Computes the value of the Rastrigin function, a non-convex function commonly used 
    in optimization problems to evaluate global optimization algorithms.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    float: The value of the Rastrigin function at the given point.
    """
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    return 20 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]))

# Analytical gradient of the bivariate Rastrigin function
def rastrigin_gradient(x):
    """
    Computes the gradient (partial derivatives) of the Rastrigin function.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    np.array: The gradient [df/dx1, df/dx2] at the given point.
    """
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    grad_x = 2 * x[0] + 20 * np.pi * np.sin(2 * np.pi * x[0])
    grad_y = 2 * x[1] + 20 * np.pi * np.sin(2 * np.pi * x[1])
    return np.array([grad_x, grad_y])

# Analytical Hessian of the bivariate Rastrigin function
def rastrigin_hessian(x):
    """
    Computes the Hessian (second partial derivatives) of the Rastrigin function.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    np.array: A 2x2 matrix representing the Hessian.
    """
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    hessian = np.array([
        [2 + 40 * np.pi**2 * np.cos(2 * np.pi * x[0]), 0],
        [0, 2 + 40 * np.pi**2 * np.cos(2 * np.pi * x[1])]
    ])
    return hessian

TEX_STRING_RASTRIGIN = r'$f(x, y) = 20 + \left(x^2 - 10\cos(2\pi x)\right) + \left(y^2 - 10\cos(2\pi y)\right)$' + '\n\n'

# MIN @ (0,0) - Known minimum value for Rastrigin function.

'''
Rosenbrock Function (non-convex) BELOW
'''

# Define the bivariate Rosenbrock function
def rosenbrock_function(x):
    """
    Computes the value of the Rosenbrock function, a non-convex optimization benchmark 
    function known for its narrow and curved valley leading to the global minimum.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    float: The value of the Rosenbrock function at the given point.
    """
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    return 5 * (x[1] - x[0]**2)**2 + (1 - x[0])**2 

# Analytical gradient of the bivariate Rosenbrock function
def rosenbrock_gradient(x):
    """
    Computes the gradient (partial derivatives) of the Rosenbrock function.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    np.array: The gradient [df/dx1, df/dx2] at the given point.
    """
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    grad_x = -20 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0])
    grad_y = 10 * (x[1] - x[0]**2)
    return np.array([grad_x, grad_y])

# Analytical Hessian of the bivariate Rosenbrock function
def rosenbrock_hessian(x):
    """
    Computes the Hessian (second partial derivatives) of the Rosenbrock function.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    np.array: A 2x2 matrix representing the Hessian.
    """
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    hessian = np.array([
        [60 * x[0]**2 - 20 * x[1] + 2, -20 * x[0]],
        [-20 * x[0], 10]
    ])
    return hessian

TEX_STRING_ROSENBROCK = r'$f(x, y) = 5(y - x^2)^2 + (1 - x)^2$' + '\n\n'

# MIN @ (1,1) - Known minimum for Rosenbrock function.

'''
Convex Function BELOW
'''

# Define a bivariate convex function
def convex_bivariate_function(x):
    """
    Computes the value of a convex quadratic function, often used in optimization
    as a simple test case for evaluating algorithms on convex problems.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    float: The value of the convex function at the given point.
    """
    return 4 * x[0]**2 + 2 * x[1]**2 + x[0] * x[1]

# Analytical gradient of the convex function
def convex_bivariate_gradient(x):
    """
    Computes the gradient (partial derivatives) of the convex quadratic function.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    np.array: The gradient [df/dx1, df/dx2] at the given point.
    """
    grad_x = 8 * x[0] + x[1]
    grad_y = 4 * x[1] + x[0]
    return np.array([grad_x, grad_y])

# Analytical Hessian of the convex function
def convex_bivariate_hessian(x):
    """
    Computes the Hessian (second partial derivatives) of the convex quadratic function.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    np.array: A 2x2 matrix representing the Hessian.
    """
    return np.array([[8, 1], 
                     [1, 4]])

TEX_STRING_CONVEX_BIVARIATE = r'$f(x, y) = 4x^2 + 2y^2 + xy$' + '\n\n'

# MIN @ (0,0) - Known minimum for this convex quadratic function.

'''
Non-Convex Function BELOW
'''

# Define the non-convex function with a shifted frame of reference in the y-direction
def non_convex_function(x):
    """
    Computes the value of a non-convex function designed to test optimization 
    algorithms on problems with complex landscapes and multiple local minima.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    float: The value of the non-convex function at the given point.
    """
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    return (x[0] - 2)**2 + 3 * (x[1] - 25)**4 - 4 * x[0] * (x[1] - 25)

# Analytical gradient of the non-convex function
def non_convex_gradient(x):
    """
    Computes the gradient (partial derivatives) of the non-convex function.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    np.array: The gradient [df/dx1, df/dx2] at the given point.
    """
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    grad_x = 2 * (x[0] - 2) - 4 * (x[1] - 25)
    grad_y = 12 * (x[1] - 25)**3 - 4 * x[0]
    return np.array([grad_x, grad_y])

# Analytical Hessian of the non-convex function
def non_convex_hessian(x):
    """
    Computes the Hessian (second partial derivatives) of the non-convex function.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    np.array: A 2x2 matrix representing the Hessian.
    """
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    hessian = np.array([
        [2, -4],
        [-4, 36 * (x[1] - 25)**2]
    ])
    return hessian

TEX_STRING_NON_CONVEX = r'$f(x, y) = (x - 2)^2 + 3(y - 25)^4 - 4x(y - 25)$' + '\n\n'

# MIN @ (-10, 10) - Known minimum for this unconstrained function.
# Constrained MIN @ (-0.823, 1.823) - Minimum with specific constraints applied.

'''
Non-differentiable Function BELOW
'''

# Define a non-differentiable function
def non_differentiable_function(x):
    """
    Computes the value of a non-differentiable function, used to test algorithms 
    capable of handling optimization problems without smooth gradients.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    float: The value of the non-differentiable function at the given point.
    """
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    return -1 / (np.ceil(np.sqrt(x[0]**2 + x[1]**2 + 0.01)))

# Placeholder for gradient of the non-differentiable function
def dummy_gradient(x):
    """
    Returns None since the function is non-differentiable.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    None
    """
    return None

# Placeholder for Hessian of the non-differentiable function
def dummy_hessian(x):
    """
    Returns None since the function is non-differentiable.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    None
    """
    return None

TEX_STRING_NODIFF = r'$f(x, y) = \frac{-1}{\lceil \sqrt{x^2 + y^2 + 0.01} \rceil}$' + '\n\n'

# MIN @ (0,0) - Known minimum for this non-differentiable function.

'''
Constraints
'''

# Define a linear constraint
def constraint_function_1(x):
    """
    Computes the value of a linear constraint function: x + y - 1 <= 0.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    float: The value of the constraint at the given point.
    """
    return x[0] + x[1] - 1

# Define a quadratic constraint
def constraint_function_2(x):
    """
    Computes the value of a quadratic constraint function: x^2 + y^2 - 4 <= 0.

    Parameters:
    x (list or np.array): A bivariate input [x1, x2].

    Returns:
    float: The value of the constraint at the given point.
    """
    return x[0]**2 + x[1]**2 - 4

# List of constraint functions
constraint_functions = [constraint_function_1, constraint_function_2]

class Particle:
    """
    Represents a particle in the swarm, used in Particle Swarm Optimization (PSO).
    Each particle represents a candidate solution and adapts its position based on
    personal and global experience.
    """

    def __init__(self, bounds, constr_funcs=None):
        """
        Initialize a particle.

        Parameters:
        - bounds (np.ndarray): Array of shape (n_dimensions, 2) specifying the min 
          and max bounds for each dimension.
        - constr_funcs (list of functions or None): List of constraint functions. 
          Each should accept a position array and return <= 0 if constraints are satisfied.
        """
        self.bounds = bounds  # Save bounds for each dimension
        
        # Randomly initialize the particle's position within the provided bounds
        self.position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        
        # Initialize velocity to zero (stationary particle)
        self.velocity = np.zeros_like(self.position)
        
        # Track the best position this particle has visited and its corresponding value
        self.best_position = self.position.copy()
        self.best_value = np.inf  # Initialize to infinity for minimization
        
        # Store constraint functions, if any
        self.constr_funcs = constr_funcs

    def evaluate(self, objective_func):
        """
        Evaluate the particle's fitness using the objective function and check constraints.

        Parameters:
        - objective_func (function): The objective function to minimize.

        Returns:
        - float: The fitness value of the particle's current position.
        """
        # Evaluate the objective function at the current position
        fitness = objective_func(self.position)
        
        # Apply constraints, if any are defined
        if self.constr_funcs:
            for constr_func in self.constr_funcs:
                if constr_func(self.position) > 0:  # Constraint violation
                    fitness = np.inf  # Penalize the particle heavily
                    break  # Exit early if a constraint is violated

        # Update personal best if the current position improves the fitness
        if fitness < self.best_value:
            self.best_value = fitness  # Update best fitness value
            self.best_position = self.position.copy()  # Update best position

        return fitness

    def update_velocity(self, global_best_position, w, c1, c2):
        """
        Update the particle's velocity based on its personal and global best positions.

        Parameters:
        - global_best_position (np.ndarray): The best position found by the entire swarm.
        - w (float): Inertia weight to control the influence of the previous velocity.
        - c1 (float): Cognitive coefficient, influencing the attraction towards the personal best.
        - c2 (float): Social coefficient, influencing the attraction towards the global best.
        """
        r1 = np.random.rand(len(self.position))  # Random factor for cognitive component
        r2 = np.random.rand(len(self.position))  # Random factor for social component
        cognitive = c1 * r1 * (self.best_position - self.position)  # Cognitive term
        social = c2 * r2 * (global_best_position - self.position)  # Social term
        
        # Update velocity as a weighted sum of inertia, cognitive, and social terms
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        """
        Update the particle's position based on its velocity and enforce boundary constraints.
        """
        self.position += self.velocity  # Update position using the current velocity
        
        # Enforce position to stay within the specified bounds
        self.position = np.clip(self.position, self.bounds[:, 0], self.bounds[:, 1])


class OptimizationAlgorithm:
    """
    Base class for optimization algorithms. Provides a general framework for implementing
    specific optimization methods such as PSO, gradient-based methods, etc.
    """

    def __init__(self, objective_func, bounds, max_iter, constr_funcs=None, analytic_solution=None):
        """
        Initialize the optimization algorithm.

        Parameters:
        - objective_func (function): The objective function to be minimized.
        - bounds (list of tuples): List of (min, max) bounds for each dimension of the search space.
        - max_iter (int): Maximum number of iterations for the optimization process.
        - constr_funcs (list of functions or None): Optional list of constraint functions.
          Each function should return <= 0 if the constraint is satisfied.
        - analytic_solution (np.ndarray or None): Known solution for the optimization problem
          (useful for benchmarking or error calculations).
        """
        self.objective_func = objective_func  # Objective function to minimize
        self.bounds = np.array(bounds)  # Bounds for the search space, converted to NumPy array
        self.max_iter = max_iter  # Maximum iterations for optimization
        self.constr_funcs = constr_funcs  # Optional constraints
        self.dimensions = len(bounds)  # Number of dimensions in the problem
        self.analytic_solution = analytic_solution  # Known solution, if provided
        
        # History dictionary to track optimization progress
        self.history = {
            'positions': [],  # Stores positions visited by the optimizer
            'values': [],  # Stores objective function values at each iteration
            'times': [],  # Tracks time taken at each iteration
            'best_positions': []  # Tracks the best positions found over time
        }

    def optimize(self):
        """
        Run the optimization algorithm. This method is intended to be overridden by subclasses
        implementing specific optimization algorithms.
        """
        pass  # Placeholder for subclass implementation

    def plot_results(self, method_name):
        """
        Plot the results of the optimization process.

        Parameters:
        - method_name (str): Name of the optimization method (e.g., "PSO", "Steepest Descent").
        """
        # Placeholder for plotting implementation, typically similar to the PSO's plot_history.
        # Would include plots of the convergence history, positions, and optionally animations.
        pass


class PSO(OptimizationAlgorithm):
    """
    Particle Swarm Optimization algorithm implementation extending the OptimizationAlgorithm base class.
    This class visualizes the optimization process using an interactive animation.
    """

    def __init__(self, objective_func, bounds, num_particles, max_iter,
                 w=0.5, c1=1.5, c2=1.5, constr_funcs=None, analytic_solution=None):
        """
        Initialize the PSO algorithm.

        Parameters:
        - objective_func (function): The objective function to minimize.
        - bounds (list of tuples): Search space bounds for each dimension [(min, max), ...].
        - num_particles (int): Number of particles in the swarm.
        - max_iter (int): Maximum number of iterations.
        - w (float): Inertia weight to balance exploration and exploitation.
        - c1 (float): Cognitive coefficient influencing personal best attraction.
        - c2 (float): Social coefficient influencing global best attraction.
        - constr_funcs (list of functions or None): Optional list of constraint functions.
        - analytic_solution (np.ndarray or None): Optional known solution for benchmarking.
        """
        super().__init__(objective_func, bounds, max_iter, constr_funcs, analytic_solution)
        self.num_particles = num_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive constant
        self.c2 = c2  # Social constant

        # Initialize swarm as a list of Particle objects
        self.swarm = [Particle(self.bounds, constr_funcs) for _ in range(self.num_particles)]

        # Initialize the global best position and value
        self.global_best_position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        self.global_best_value = np.inf

    def optimize(self):
        """
        Run the PSO algorithm with visualization of the optimization process.

        Creates an interactive animation showing particle movement, the global best position, 
        and constraint boundaries (if applicable) in a 2D space.
        """
        start_time = time.time()  # Start timer for tracking elapsed time

        # Configure the plot layout
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 5, 1.2])

        # Left subplot for static text information
        ax_text_left = fig.add_subplot(gs[0])
        ax_text_left.axis('off')  # Hide axis

        # Right subplot for dynamic text updates
        ax_text_right = fig.add_subplot(gs[2])
        ax_text_right.axis('off')  # Hide axis

        # Center subplot for the animation
        ax = fig.add_subplot(gs[1])
        ax.set_box_aspect(1)  # Ensure equal aspect ratio

        # Adjust subplot layout
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        if self.dimensions == 2:
            # Define the grid for plotting the objective function
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)

            # Compute objective function values over the grid
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(X.ravel(), Y.ravel())])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')  # Contour plot
            plt.colorbar(contour, ax=ax)  # Add color bar for reference

            # Plot constraints if defined
            if self.constr_funcs:
                for constr_func in self.constr_funcs:
                    # Evaluate the constraint function over the grid
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.ravel(), Y.ravel())])
                    constraint_values = constraint_values.reshape(X.shape)
                    # Plot the zero contour of the constraint
                    ax.contour(X, Y, constraint_values, levels=[0], colors='white', linewidths=2, linestyles='dashed')

        # Static text information
        static_info = 'Objective Function:\n'
        static_info += TEX_STRING

        if self.constr_funcs:
            static_info += 'Constraints:\n'
            static_info += r'$x + y - 1 \leq 0$' + '\n'
            static_info += r'$x^2 + y^2 - 4 \leq 0$' + '\n\n'

        # Display parameters
        static_info += 'Parameters:\n'
        static_info += f'Number of Particles: {self.num_particles}\n'
        static_info += f'Bounds: {self.bounds.tolist()}\n'
        static_info += f'$w$: {self.w}\n'
        static_info += f'$c1$: {self.c1}\n'
        static_info += f'$c2$: {self.c2}\n\n'

        if self.analytic_solution is not None:
            static_info += 'Analytic Solution:\n'
            static_info += f'{np.round(self.analytic_solution, 4)}\n'

        # Add static information to the left text subplot
        ax_text_left.text(0, 0.5, static_info, fontsize=10, va='center')

        # Initialize scatter plots for particles and best positions
        scat = ax.scatter([], [], c='red', label='Particles', s=30)
        best_scat = ax.scatter([], [], c='blue', label='Global Best', s=50)

        if self.analytic_solution is not None:
            analytic_scat = ax.scatter(self.analytic_solution[0], self.analytic_solution[1],
                                       c='green', label='Analytic Solution', marker='x', s=100)

        # Configure legend
        ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

        def init():
            """
            Initialize scatter plots for the animation.
            """
            empty_data = np.empty((0, 2))
            scat.set_offsets(empty_data)
            best_scat.set_offsets(empty_data)
            return scat, best_scat

        def update(frame):
            """
            Update particle positions and global best during the animation.

            Parameters:
            - frame (int): Current frame (iteration number).
            """
            current_time = time.time() - start_time
            for particle in self.swarm:
                fitness = particle.evaluate(self.objective_func)
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = particle.position.copy()

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()

            # Update scatter plot positions
            positions = np.array([particle.position for particle in self.swarm])
            scat.set_offsets(positions)
            best_scat.set_offsets([self.global_best_position])

            # Update dynamic text
            iteration_number = frame + 1
            dynamic_info = f'Iteration: {iteration_number}\n'
            dynamic_info += f'Time Elapsed: {current_time:.2f}s\n'
            dynamic_info += f'Best Position: {np.round(self.global_best_position, 4)}\n'
            dynamic_info += f'Best Value: {self.global_best_value:.6f}\n'
            ax_text_right.clear()
            ax_text_right.axis('off')
            ax_text_right.text(0.05, 0.5, dynamic_info, fontsize=10, va='center')

            # Save optimization history
            self.history['positions'].append(positions.copy())
            self.history['best_positions'].append(self.global_best_position.copy())
            self.history['values'].append(self.global_best_value)
            self.history['times'].append(current_time)

            return scat, best_scat

        # Create and display animation
        ani = animation.FuncAnimation(fig, update, frames=self.max_iter, init_func=init,
                                      blit=False, interval=200, repeat=False)
        plt.tight_layout()
        plt.show()

        # Animate the best position over iterations
        self.animate_best_position()


    def animate_best_position(self):
        """
        Create an animation showing the movement of the global best position over iterations.

        This visualization plots the global best path against the objective function's contour
        and optionally overlays constraint boundaries if they are defined.
        """
        if self.dimensions == 2:
            # Set up the plot for the animation
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            # Generate grid for plotting the objective function
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(np.ravel(X), np.ravel(Y))])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')  # Contour plot of the objective function
            plt.colorbar(contour, ax=ax)

            # Plot constraint boundaries, if any
            if self.constr_funcs:
                constraint_handles = []
                constraint_labels = []
                constraint_colors = ['white', 'yellow']  # Colors for constraints
                for idx, constr_func in enumerate(self.constr_funcs):
                    # Evaluate the constraint function over the grid
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
                    constraint_values = constraint_values.reshape(X.shape)
                    # Plot the zero level contour (constraint boundary)
                    contour_set = ax.contour(X, Y, constraint_values, levels=[0],
                                            colors=constraint_colors[idx % len(constraint_colors)],
                                            linewidths=2, linestyles='dashed')
                    # Store handles and labels for the legend
                    constraint_handles.append(contour_set.collections[0])
                    constraint_labels.append(f'Constraint {idx+1}')

            # Set axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Prepare data for animation
            best_positions = np.array(self.history['best_positions'])

            # Initialize scatter plot for the global best path
            best_scat, = ax.plot([], [], 'bo-', label='Global Best Path')  # Blue line and dots
            if self.analytic_solution is not None:
                ax.plot(self.analytic_solution[0], self.analytic_solution[1],
                        'gx', label='Analytic Solution', markersize=10, markeredgewidth=2)  # Green 'x'

            # Adjust legend to include constraints
            handles, labels = ax.get_legend_handles_labels()
            if self.constr_funcs:
                handles += constraint_handles
                labels += constraint_labels
            ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

            def init():
                """Initialize animation by clearing the global best path."""
                best_scat.set_data([], [])
                return best_scat,

            def update(frame):
                """Update animation with global best positions up to the current frame."""
                xdata = best_positions[:frame+1, 0]
                ydata = best_positions[:frame+1, 1]
                best_scat.set_data(xdata, ydata)
                return best_scat,

            # Create and display the animation
            ani = animation.FuncAnimation(fig, update, frames=len(best_positions),
                                        init_func=init, blit=True, interval=200, repeat=False)
            plt.title('Global Best Position Over Iterations')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.tight_layout()
            plt.show()
        else:
            print("Animation of best position over iterations is only available for 2D problems.")

    def plot_history(self):
        """
        Plot the history of the best objective function values and best positions over iterations.

        Creates two sets of plots:
        1. Best objective function value over iterations.
        2. Best position for each dimension over iterations.
        """
        iterations = range(len(self.history['values']))

        # Plot Best Objective Function Value Over Iterations
        plt.figure()
        plt.plot(iterations, self.history['values'], marker='o')
        plt.title('Best Objective Function Value Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Value')
        plt.grid(True)
        plt.show()

        # Plot Best Position Over Iterations for Each Dimension
        for i in range(self.dimensions):
            plt.figure()
            positions = [pos[i] for pos in self.history['best_positions']]
            plt.plot(iterations, positions, marker='o', label=f'Dimension {i+1}')
            plt.title(f'Best Position Dimension {i+1} Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel(f'Best Position Dimension {i+1}')
            plt.grid(True)
            plt.legend()
            plt.show()


class GradientDescent(OptimizationAlgorithm):
    """
    Gradient Descent optimization algorithm, with support for visualization and constraint handling.
    """

    def __init__(self, objective_func, bounds, max_iter, constr_funcs=None, analytic_solution=None, alpha=0.025):
        """
        Initialize the Gradient Descent optimizer.

        Parameters:
        - objective_func (function): The objective function to minimize.
        - bounds (list of tuples): Bounds for the search space [(min, max), ...].
        - max_iter (int): Maximum number of iterations.
        - constr_funcs (list of functions or None): Optional list of constraint functions.
        - analytic_solution (np.ndarray or None): Known solution for benchmarking.
        - alpha (float): Step size for the gradient descent updates.
        """
        super().__init__(objective_func, bounds, max_iter, constr_funcs, analytic_solution)
        self.alpha = alpha  # Step size

    def optimize(self):
        """
        Run the Gradient Descent algorithm with visualization of the optimization process.

        Includes dynamic updates of the current point and objective value, with optional constraint handling.
        """
        start_time = time.time()  # Start timer

        # Initialize the starting point randomly within bounds
        x_current = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        self.history['positions'].append(x_current.copy())
        self.history['values'].append(self.objective_func(x_current))
        self.history['times'].append(0)

        # Configure plot layout
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 5, 1.2])

        # Static text on the left
        ax_text_left = fig.add_subplot(gs[0])
        ax_text_left.axis('off')

        # Dynamic text on the right
        ax_text_right = fig.add_subplot(gs[2])
        ax_text_right.axis('off')

        # Contour plot in the middle
        ax = fig.add_subplot(gs[1])
        ax.set_box_aspect(1)

        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        if self.dimensions == 2:
            # Generate grid for contour plot
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 200)
            y = np.linspace(y_min, y_max, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax)

            # Plot constraint boundaries if present
            if self.constr_funcs:
                for idx, constr_func in enumerate(self.constr_funcs):
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
                    constraint_values = constraint_values.reshape(X.shape)
                    ax.contour(X, Y, constraint_values, levels=[0], colors='white', linewidths=2, linestyles='dashed')

        # Static information for display
        static_info = 'Gradient Descent Optimization\n\nObjective Function:\n'
        static_info += TEX_STRING
        if self.constr_funcs:
            static_info += 'Constraints:\n'
            static_info += r'$x + y - 1 \leq 0$' + '\n'
            static_info += r'$x^2 + y^2 - 4 \leq 0$' + '\n\n'
        static_info += f'Parameters:\nStep Size (alpha): {self.alpha}\n\n'
        if self.analytic_solution is not None:
            static_info += f'Analytic Solution:\n{np.round(self.analytic_solution, 4)}\n'
        ax_text_left.text(0, 0.5, static_info, fontsize=10, va='center')

        # Scatter plot for the current point
        point_scat = ax.scatter([], [], c='red', label='Current Point', s=50)
        if self.analytic_solution is not None:
            ax.scatter(self.analytic_solution[0], self.analytic_solution[1],
                       c='green', label='Analytic Solution', marker='x', s=100)
        ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

        def init():
            """Initialize scatter plot."""
            point_scat.set_offsets(np.empty((0, 2)))
            return point_scat,

        def gradient(x):
            """Compute the numerical gradient of the objective function."""
            h = 1e-5  # Small step for finite difference
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_step = x.copy()
                x_step[i] += h
                grad[i] = (self.objective_func(x_step) - self.objective_func(x)) / h
            return grad

        def project_onto_constraints(x):
            """Project a point onto the feasible region defined by constraints."""
            from scipy.optimize import minimize

            def distance(y):
                return np.linalg.norm(y - x)

            cons = [{'type': 'ineq', 'fun': constr_func} for constr_func in self.constr_funcs]
            bounds = self.bounds.tolist()
            res = minimize(distance, x, bounds=bounds, constraints=cons)
            return res.x if res.success else x

        def update(frame):
            """Update the current point and visualize its movement."""
            nonlocal x_current
            current_time = time.time() - start_time

            grad = gradient(x_current)
            x_next = x_current - self.alpha * grad

            # Apply constraints if defined
            if self.constr_funcs:
                x_next = project_onto_constraints(x_next)

            x_current = x_next
            value = self.objective_func(x_current)

            # Update history
            self.history['positions'].append(x_current.copy())
            self.history['values'].append(value)
            self.history['times'].append(current_time)

            # Update scatter plot
            point_scat.set_offsets([x_current])

            # Update dynamic text
            iteration_number = frame + 1
            dynamic_info = f'Iteration: {iteration_number}\nTime Elapsed: {current_time:.2f}s\n'
            dynamic_info += f'Current Position: {np.round(x_current, 4)}\nObjective Value: {value:.6f}\n'
            ax_text_right.clear()
            ax_text_right.axis('off')
            ax_text_right.text(0.05, 0.5, dynamic_info, fontsize=10, va='center')

            return point_scat,

        # Animate the optimization process
        ani = animation.FuncAnimation(fig, update, frames=self.max_iter, init_func=init,
                                      blit=False, interval=200, repeat=False)
        plt.tight_layout()
        plt.show()

        # Plot results
        self.plot_results('Gradient Descent')


class NewtonMethod(OptimizationAlgorithm):
    """
    Newton's Method optimization algorithm with visualization and constraint handling.
    """

    def optimize(self):
        """
        Run Newton's Method to minimize the objective function and visualize the process.

        This implementation uses `scipy.optimize.minimize` with the `trust-constr` method, which supports
        constraints, gradients, and Hessians for efficient optimization.
        """
        start_time = time.time()  # Start timer

        # Initial guess within bounds
        x_current = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        self.history['positions'].append(x_current.copy())
        self.history['values'].append(self.objective_func(x_current))
        self.history['times'].append(0)

        # Configure plot layout
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 5, 1.2])

        # Left panel for static text
        ax_text_left = fig.add_subplot(gs[0])
        ax_text_left.axis('off')

        # Right panel for dynamic text
        ax_text_right = fig.add_subplot(gs[2])
        ax_text_right.axis('off')

        # Center panel for contour plot
        ax = fig.add_subplot(gs[1])
        ax.set_box_aspect(1)

        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        if self.dimensions == 2:
            # Generate grid for contour plot
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 200)
            y = np.linspace(y_min, y_max, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax)

            # Plot constraint boundaries, if any
            if self.constr_funcs:
                for idx, constr_func in enumerate(self.constr_funcs):
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
                    constraint_values = constraint_values.reshape(X.shape)
                    ax.contour(X, Y, constraint_values, levels=[0], colors='white', linewidths=2, linestyles='dashed')

        # Static information display
        static_info = "Newton's Method Optimization\n\nObjective Function:\n"
        static_info += TEX_STRING
        if self.constr_funcs:
            static_info += 'Constraints:\n'
            static_info += r'$x + y - 1 \leq 0$' + '\n'
            static_info += r'$x^2 + y^2 - 4 \leq 0$' + '\n\n'
        if self.analytic_solution is not None:
            static_info += f'Analytic Solution:\n{np.round(self.analytic_solution, 4)}\n'
        ax_text_left.text(0, 0.5, static_info, fontsize=10, va='center')

        # Scatter plot for current point
        point_scat = ax.scatter([], [], c='red', label='Current Point', s=50)
        if self.analytic_solution is not None:
            ax.scatter(self.analytic_solution[0], self.analytic_solution[1],
                       c='green', label='Analytic Solution', marker='x', s=100)
        ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

        # Prepare constraints and bounds for `trust-constr`
        cons = [{'type': 'ineq', 'fun': constr_func} for constr_func in self.constr_funcs] if self.constr_funcs else None
        bounds = self.bounds.tolist()

        # Callback to track progress
        self.iteration = 0
        self.positions = [x_current.copy()]
        self.times = [0]

        def callback(xk, state=None):
            """Store iteration data for visualization."""
            self.iteration += 1
            current_time = time.time() - start_time
            value = self.objective_func(xk)

            self.history['positions'].append(xk.copy())
            self.history['values'].append(value)
            self.history['times'].append(current_time)
            self.positions.append(xk.copy())
            self.times.append(current_time)

        # Run optimization using Newton's Method
        res = minimize(
            self.objective_func,
            x_current,
            method='trust-constr',
            jac=objective_gradient,  # Gradient of the objective
            hess=objective_hessian,  # Hessian of the objective
            constraints=cons,
            bounds=bounds,
            callback=callback,
            options={'maxiter': self.max_iter, 'verbose': 0}
        )

        # Extract optimization history
        positions = np.array(self.history['positions'])

        def init():
            """Initialize scatter plot."""
            point_scat.set_offsets(np.empty((0, 2)))
            return point_scat,

        def update(frame):
            """Update scatter plot and text information dynamically."""
            if frame < len(positions):
                xk = positions[frame]
                time_elapsed = self.history['times'][frame]
                value = self.history['values'][frame]

                point_scat.set_offsets([xk])

                # Update dynamic text
                iteration_number = frame + 1
                dynamic_info = f'Iteration: {iteration_number}\n'
                dynamic_info += f'Time Elapsed: {time_elapsed:.2f}s\n'
                dynamic_info += f'Current Position: {np.round(xk, 4)}\n'
                dynamic_info += f'Objective Value: {value:.6f}\n'
                ax_text_right.clear()
                ax_text_right.axis('off')
                ax_text_right.text(0.05, 0.5, dynamic_info, fontsize=10, va='center')

            return point_scat,

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(positions), init_func=init,
                                      blit=False, interval=500, repeat=False)
        plt.tight_layout()
        plt.show()

        # Plot results
        self.plot_results("Newton's Method")


class BFGSMethod(OptimizationAlgorithm):
    """
    BFGS optimization algorithm with visualization and constraint handling.
    """

    def optimize(self):
        """
        Run the BFGS optimization method to minimize the objective function and visualize the process.

        Supports both constrained and unconstrained optimization using appropriate algorithms
        (`SLSQP` for constrained problems, `L-BFGS-B` for unconstrained).
        """
        start_time = time.time()  # Start timer

        # Initialize starting point randomly within bounds
        x_current = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        self.history['positions'].append(x_current.copy())
        self.history['values'].append(self.objective_func(x_current))
        self.history['times'].append(0)

        # Configure plot layout
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 5, 1.2])

        # Left panel for static text
        ax_text_left = fig.add_subplot(gs[0])
        ax_text_left.axis('off')

        # Right panel for dynamic text
        ax_text_right = fig.add_subplot(gs[2])
        ax_text_right.axis('off')

        # Center panel for contour plot
        ax = fig.add_subplot(gs[1])
        ax.set_box_aspect(1)

        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        if self.dimensions == 2:
            # Generate grid for contour plot
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 200)
            y = np.linspace(y_min, y_max, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax)

            # Plot constraint boundaries, if any
            if self.constr_funcs:
                for idx, constr_func in enumerate(self.constr_funcs):
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
                    constraint_values = constraint_values.reshape(X.shape)
                    ax.contour(X, Y, constraint_values, levels=[0], colors='white', linewidths=2, linestyles='dashed')

        # Static information display
        static_info = 'BFGS Optimization\n\nObjective Function:\n'
        static_info += TEX_STRING
        if self.constr_funcs:
            static_info += 'Constraints:\n'
            static_info += r'$x + y - 1 \leq 0$' + '\n'
            static_info += r'$x^2 + y^2 - 4 \leq 0$' + '\n\n'
        if self.analytic_solution is not None:
            static_info += f'Analytic Solution:\n{np.round(self.analytic_solution, 4)}\n'
        ax_text_left.text(0, 0.5, static_info, fontsize=10, va='center')

        # Scatter plot for current point
        point_scat = ax.scatter([], [], c='red', label='Current Point', s=50)
        if self.analytic_solution is not None:
            ax.scatter(self.analytic_solution[0], self.analytic_solution[1],
                       c='green', label='Analytic Solution', marker='x', s=100)
        ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

        # Prepare constraints and bounds for `minimize`
        cons = [{'type': 'ineq', 'fun': constr_func} for constr_func in self.constr_funcs] if self.constr_funcs else None
        bounds = self.bounds.tolist()

        # Initialize iteration tracker
        self.iteration = 0
        self.positions = [x_current.copy()]
        self.times = [0]

        # Define callback to store iteration data
        def callback(xk):
            self.iteration += 1
            current_time = time.time() - start_time
            value = self.objective_func(xk)

            self.history['positions'].append(xk.copy())
            self.history['values'].append(value)
            self.history['times'].append(current_time)
            self.positions.append(xk.copy())
            self.times.append(current_time)

        # Select method based on constraints
        method = 'SLSQP' if self.constr_funcs else 'L-BFGS-B'

        # Run optimization
        res = minimize(
            self.objective_func,
            x_current,
            method=method,
            jac=objective_gradient,  # Gradient of the objective
            constraints=cons,
            bounds=bounds,
            callback=callback,
            options={'maxiter': self.max_iter, 'disp': False}
        )

        # Extract optimization history
        positions = np.array(self.history['positions'])

        def init():
            """Initialize scatter plot."""
            point_scat.set_offsets(np.empty((0, 2)))
            return point_scat,

        def update(frame):
            """Update scatter plot and text information dynamically."""
            if frame < len(positions):
                xk = positions[frame]
                time_elapsed = self.history['times'][frame]
                value = self.history['values'][frame]

                point_scat.set_offsets([xk])

                # Update dynamic text
                iteration_number = frame + 1
                dynamic_info = f'Iteration: {iteration_number}\n'
                dynamic_info += f'Time Elapsed: {time_elapsed:.2f}s\n'
                dynamic_info += f'Current Position: {np.round(xk, 4)}\n'
                dynamic_info += f'Objective Value: {value:.6f}\n'
                ax_text_right.clear()
                ax_text_right.axis('off')
                ax_text_right.text(0.05, 0.5, dynamic_info, fontsize=10, va='center')

            return point_scat,

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(positions), init_func=init,
                                      blit=False, interval=500, repeat=False)
        plt.tight_layout()
        plt.show()

        # Plot results
        self.plot_results('BFGS Method')


def run_comparison_mode(objective_func, bounds, max_iter, constr_funcs, analytic_solution):
    """
    Run the comparison mode to evaluate and compare the performance of different optimization algorithms.

    Parameters:
    - objective_func (function): The objective function to minimize.
    - bounds (list of tuples): Search space bounds for each dimension [(min, max), ...].
    - max_iter (int): Maximum number of iterations for each algorithm.
    - constr_funcs (list of functions or None): Optional list of constraint functions.
    - analytic_solution (np.ndarray or None): Known solution for benchmarking, if available.
    """
    # Instantiate optimization algorithms
    pso = PSO(
        objective_func=objective_func,
        bounds=bounds,
        num_particles=30,  # Number of particles in PSO
        max_iter=max_iter,
        w=0.5,  # Inertia weight for PSO
        c1=1.5,  # Cognitive coefficient for PSO
        c2=1.5,  # Social coefficient for PSO
        constr_funcs=constr_funcs,
        analytic_solution=analytic_solution
    )

    gd = GradientDescent(
        objective_func=objective_func,
        bounds=bounds,
        max_iter=max_iter,
        constr_funcs=constr_funcs,
        analytic_solution=analytic_solution
    )

    newton = NewtonMethod(
        objective_func=objective_func,
        bounds=bounds,
        max_iter=max_iter,
        constr_funcs=constr_funcs,
        analytic_solution=analytic_solution
    )

    bfgs = BFGSMethod(
        objective_func=objective_func,
        bounds=bounds,
        max_iter=max_iter,
        constr_funcs=constr_funcs,
        analytic_solution=analytic_solution
    )

    # Execute each algorithm and print progress
    print("Running PSO...")
    pso.optimize()
    print("Running Gradient Descent...")
    gd.optimize()
    print("Running Newton's Method...")
    newton.optimize()
    print("Running BFGS Method...")
    bfgs.optimize()


# Main execution
if __name__ == "__main__":
    """
    Main execution loop for the optimization script.
    Prompts the user to select the optimization function, mode, and constraint preferences,
    and then runs the chosen optimization mode (PSO or comparison) on the selected function.
    """

    # Function mapping
    functions = {
        1: ("Rastrigin", rastrigin_function, rastrigin_gradient, rastrigin_hessian, TEX_STRING_RASTRIGIN),
        2: ("Rosenbrock", rosenbrock_function, rosenbrock_gradient, rosenbrock_hessian, TEX_STRING_ROSENBROCK),
        3: ("Convex", convex_bivariate_function, convex_bivariate_gradient, convex_bivariate_hessian, TEX_STRING_CONVEX_BIVARIATE),
        4: ("Non-Convex", non_convex_function, non_convex_gradient, non_convex_hessian, TEX_STRING_NON_CONVEX),
        5: ("Non-Differentiable Function", non_differentiable_function, dummy_gradient, dummy_hessian, TEX_STRING_NODIFF)
    }

    # Prompt user to select the function
    print("Select the function to optimize:")
    for key, value in functions.items():
        print(f"{key}: {value[0]}")

    selected_function = int(input("Enter the function number: "))
    if selected_function not in functions:
        print("Invalid function number. Exiting.")
        exit()

    # Set the selected function
    selected_func_name, objective_function, objective_gradient, objective_hessian, TEX_STRING = functions[selected_function]
    print(f"Selected function: {selected_func_name}")

    # Define bounds for each dimension
    bounds = [(-10, 10), (-10, 10)]

    # Ask user for analytic solution
    analytic_solution_input = input("Enter the analytic solution coordinates separated by commas (or press Enter to skip): ")
    if analytic_solution_input.strip():
        analytic_solution = np.array([float(coord) for coord in analytic_solution_input.split(',')])
    else:
        analytic_solution = None

    # Ask user for the optimization mode
    mode_input = input("Enter '1' to run PSO mode or '2' for comparison mode: ")

    # Ask user if they want constrained or unconstrained optimization
    constraint_input = input("Enter '1' for constrained optimization or '2' for unconstrained optimization: ")
    if constraint_input.strip() == '1':
        constr_funcs = constraint_functions
    elif constraint_input.strip() == '2':
        constr_funcs = None
    else:
        print("Invalid input. Exiting.")
        exit()

    # Execute the selected mode
    if mode_input.strip() == '1':
        # Run PSO mode
        pso = PSO(
            objective_function,
            bounds,
            num_particles=30,
            max_iter=50,
            w=0.5,
            c1=1.5,
            c2=1.5,
            constr_funcs=constr_funcs,
            analytic_solution=analytic_solution
        )
        pso.optimize()
    elif mode_input.strip() == '2':
        # Run comparison mode
        run_comparison_mode(
            objective_function,
            bounds,
            max_iter=50,
            constr_funcs=constr_funcs,
            analytic_solution=analytic_solution
        )
    else:
        print("Invalid input. Exiting.")
