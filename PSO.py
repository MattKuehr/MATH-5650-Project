# Author: Matt Kuehr / mck0063@auburn.edu

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.optimize import minimize

'''
Rastrigin Function (not-convex) BELOW
'''

# Define the bivariate Rastrigin function
def rastrigin_function(x):
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    return 20 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]))

# Analytical gradient of the bivariate Rastrigin function
def rastrigin_gradient(x):
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    grad_x = 2 * x[0] + 20 * np.pi * np.sin(2 * np.pi * x[0])
    grad_y = 2 * x[1] + 20 * np.pi * np.sin(2 * np.pi * x[1])
    return np.array([grad_x, grad_y])

# Analytical Hessian of the bivariate Rastrigin function
def rastrigin_hessian(x):
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    hessian = np.array([
        [2 + 40 * np.pi**2 * np.cos(2 * np.pi * x[0]), 0],
        [0, 2 + 40 * np.pi**2 * np.cos(2 * np.pi * x[1])]
    ])
    return hessian

TEX_STRING_RASTRIGIN = r'$f(x, y) = 20 + \left(x^2 - 10\cos(2\pi x)\right) + \left(y^2 - 10\cos(2\pi y)\right)$' + '\n\n'

# MIN @ (0,0)


'''
Rosenbrock Function (not-convex) BELOW
'''

# Define the bivariate Rosenbrock function
def rosenbrock_function(x):
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    return 5 * (x[1] - x[0]**2)**2 + (1 - x[0])**2 

# Analytical gradient of the bivariate Rosenbrock function
def rosenbrock_gradient(x):
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    grad_x = -20 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0])
    grad_y = 10 * (x[1] - x[0]**2)
    return np.array([grad_x, grad_y])

# Analytical Hessian of the bivariate Rosenbrock function
def rosenbrock_hessian(x):
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    hessian = np.array([
        [60 * x[0]**2 - 20 * x[1] + 2, -20 * x[0]],
        [-20 * x[0], 10]
    ])
    return hessian

TEX_STRING_ROSENBROCK = r'$f(x, y) = 5(y - x^2)^2 + (1 - x)^2$' + '\n\n'

# MIN @ (1,1)

'''
Convex Function BELOW
'''

# Define a bivariate convex function
def convex_bivariate_function(x):
    return 4 * x[0]**2 + 2 * x[1]**2 + x[0] * x[1]

# Analytical gradient of the convex function
def convex_bivariate_gradient(x):
    grad_x = 8 * x[0] + x[1]
    grad_y = 4 * x[1] + x[0]
    return np.array([grad_x, grad_y])

# Analytical Hessian of the convex function
def convex_bivariate_hessian(x):
    return np.array([[8, 1], 
                     [1, 4]])

TEX_STRING_CONVEX_BIVARIATE = r'$f(x, y) = 4x^2 + 2y^2 + xy$' + '\n\n'

# MIN @ (0,0)

'''
Non-Convex Function BELOW
'''

# Define the non-convex function with the frame of reference shifted upward in the y-direction
def non_convex_function(x):
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    return (x[0] - 2)**2 + 3 * (x[1] - 25)**4 - 4 * x[0] * (x[1] - 25)

# Analytical gradient of the non-convex function
def non_convex_gradient(x):
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    grad_x = 2 * (x[0] - 2) - 4 * (x[1] - 25)
    grad_y = 12 * (x[1] - 25)**3 - 4 * x[0]
    return np.array([grad_x, grad_y])

# Analytical Hessian of the non-convex function
def non_convex_hessian(x):
    x = np.array(x)  # Convert input to a NumPy array
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    hessian = np.array([
        [2, -4],
        [-4, 36 * (x[1] - 25)**2]
    ])
    return hessian

TEX_STRING_NON_CONVEX = r'$f(x, y) = (x - 2)^2 + 3(y - 25)^4 - 4x(y - 25)$' + '\n\n'

# MIN @ (-10, 10)

# Constrained MIN @ (-0.823, 1.823)

'''
Non-differentiable Function BELOW
'''

def non_differentiable_function(x):
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("This function is bivariate; input must have exactly two variables.")
    return -1 / (np.ceil(np.sqrt(x[0]**2 + x[1]**2 + 0.01)))  

def dummy_gradient(x):
    return None

def dummy_hessian(x):
    return None

TEX_STRING_NODIFF = r'$f(x, y) = \frac{-1}{\lceil \sqrt{x^2 + y^2 + 0.01} \rceil}$' + '\n\n'

# MIN @ (0,0)


'''
Constraints
'''

def constraint_function_1(x):
    # Example constraint: x + y - 1 <= 0
    return x[0] + x[1] - 1

def constraint_function_2(x):
    # Example constraint: x^2 + y^2 - 4 <= 0
    return x[0]**2 + x[1]**2 - 4

# List of constraint functions
constraint_functions = [constraint_function_1, constraint_function_2]

class Particle:
    """
    Represents a particle in the swarm.
    """
    def __init__(self, bounds, constr_funcs=None):
        """
        Initialize a particle.

        Parameters:
        - bounds: np.ndarray
            An array of shape (n_dimensions, 2) specifying the min and max bounds for each dimension.
        - constr_funcs: list of functions or None
            List of constraint functions. Each should accept a position array and return <= 0 if constraints are satisfied.
        """
        self.bounds = bounds  # bounds is already a NumPy array
        # Initialize position within bounds
        self.position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        # Initialize velocity to zero
        self.velocity = np.zeros_like(self.position)
        # Personal best position and value
        self.best_position = self.position.copy()
        self.best_value = np.inf
        # Constraint functions
        self.constr_funcs = constr_funcs

    def evaluate(self, objective_func):
        """
        Evaluate the particle's fitness using the objective function.

        Parameters:
        - objective_func: function
            The objective function to minimize.

        Returns:
        - fitness: float
            The fitness value of the particle's current position.
        """
        # Check constraints
        fitness = objective_func(self.position)
        if self.constr_funcs:
            for constr_func in self.constr_funcs:
                if constr_func(self.position) > 0:
                    # Penalize positions violating constraints
                    fitness = np.inf
                    break  # No need to check other constraints if one is violated

        # Update personal best if current fitness is better
        if fitness < self.best_value:
            self.best_value = fitness
            self.best_position = self.position.copy()

        return fitness

    def update_velocity(self, global_best_position, w, c1, c2):
        """
        Update the particle's velocity based on personal and global best positions.

        Parameters:
        - global_best_position: np.ndarray
            The best position found by the swarm.
        - w: float
            Inertia weight.
        - c1: float
            Cognitive coefficient.
        - c2: float
            Social coefficient.
        """
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        """
        Update the particle's position based on its velocity and apply bounds.
        """
        self.position += self.velocity
        # Apply bounds
        self.position = np.clip(self.position, self.bounds[:, 0], self.bounds[:, 1])

class OptimizationAlgorithm:
    """
    Base class for optimization algorithms.
    """
    def __init__(self, objective_func, bounds, max_iter, constr_funcs=None, analytic_solution=None):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.max_iter = max_iter
        self.constr_funcs = constr_funcs
        self.dimensions = len(bounds)
        self.analytic_solution = analytic_solution
        self.history = {'positions': [], 'values': [], 'times': [], 'best_positions': []}

    def optimize(self):
        """
        Run the optimization algorithm.
        """
        pass  # To be implemented by subclasses

    def plot_results(self, method_name):
        """
        Plot the optimization results.

        Parameters:
        - method_name: str
            Name of the optimization method.
        """
        # Similar plotting code as in PSO
        pass  # Implementation similar to PSO's plot_history

class PSO(OptimizationAlgorithm):
    """
    Particle Swarm Optimization algorithm.
    """
    def __init__(self, objective_func, bounds, num_particles, max_iter,
                 w=0.5, c1=1.5, c2=1.5, constr_funcs=None, analytic_solution=None):
        super().__init__(objective_func, bounds, max_iter, constr_funcs, analytic_solution)
        self.num_particles = num_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive constant
        self.c2 = c2  # Social constant
        # Initialize swarm
        self.swarm = [Particle(self.bounds, constr_funcs) for _ in range(self.num_particles)]
        # Initialize global best
        self.global_best_position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        self.global_best_value = np.inf

    def optimize(self):
        """
        Run the PSO algorithm and visualize the results.
        """
        start_time = time.time()

        # Adjust figure layout
        fig = plt.figure(figsize=(15, 6)) 
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 5, 1.2])

        # Left axis for static text
        ax_text_left = fig.add_subplot(gs[0])
        ax_text_left.axis('off')  # Hide the axis

        # Right axis for dynamic text
        ax_text_right = fig.add_subplot(gs[2])
        ax_text_right.axis('off')  # Hide the axis

        # Middle axis for the animation
        ax = fig.add_subplot(gs[1])
        ax.set_box_aspect(1)  # Set aspect ratio to 1:1

        # Adjust the subplot parameters to reduce left margin
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        if self.dimensions == 2:
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(np.ravel(X), np.ravel(Y))])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax)

            # Plot the constraint boundaries
            if self.constr_funcs:
                for idx, constr_func in enumerate(self.constr_funcs):
                    # Evaluate the constraint function over the grid
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
                    constraint_values = constraint_values.reshape(X.shape)
                    # Plot the zero level contour of the constraint function
                    ax.contour(X, Y, constraint_values, levels=[0], colors='white', linewidths=2, linestyles='dashed')

        # Prepare static information text
        static_info = ''

        # Objective function representation
        static_info += 'Objective Function:\n'
        static_info += TEX_STRING

        # Constraint functions representation
        if self.constr_funcs:
            static_info += 'Constraints:\n'
            for idx, constr_func in enumerate(self.constr_funcs, start=1):
                
                # Assuming we have a way to display the constraint functions as strings
                # For this example, we'll hardcode the constraint expressions
                if idx == 1:
                    static_info += r'$x + y - 1 \leq 0$' + '\n'
                elif idx == 2:
                    static_info += r'$x^2 + y^2 - 4 \leq 0$' + '\n'
            static_info += '\n'

        # Add parameters and settings
        static_info += 'Parameters:\n'
        static_info += f'Number of Particles: {self.num_particles}\n'
        static_info += f'Bounds: {self.bounds.tolist()}\n'
        static_info += f'$w$: {self.w}\n'
        static_info += f'$c1$: {self.c1}\n'
        static_info += f'$c2$: {self.c2}\n\n'

        # Analytic solution
        if self.analytic_solution is not None:
            static_info += 'Analytic Solution:\n'
            static_info += f'{np.round(self.analytic_solution, 4)}\n'

        ax_text_left.text(0, 0.5, static_info, fontsize=10, va='center')

        scat = ax.scatter([], [], c='red', label='Particles', s=30)
        best_scat = ax.scatter([], [], c='blue', label='Global Best', s=50)
        if self.analytic_solution is not None:
            analytic_scat = ax.scatter(self.analytic_solution[0], self.analytic_solution[1],
                                       c='green', label='Analytic Solution', marker='x', s=100)
        # Adjust legend
        ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

        def init():
            # Initialize scatter plots with empty data
            empty_data = np.empty((0, 2))
            scat.set_offsets(empty_data)
            best_scat.set_offsets(empty_data)
            return scat, best_scat

        def update(frame):
            current_time = time.time() - start_time
            for particle in self.swarm:
                fitness = particle.evaluate(self.objective_func)
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = particle.position.copy()

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()

            positions = np.array([particle.position for particle in self.swarm])
            scat.set_offsets(positions)

            best_scat.set_offsets([self.global_best_position])

            # Update dynamic text on the right
            iteration_number = frame + 1  # Adjust iteration count
            dynamic_info = f'Iteration: {iteration_number}\n'
            dynamic_info += f'Time Elapsed: {current_time:.2f}s\n'
            dynamic_info += f'Best Position: {np.round(self.global_best_position, 4)}\n'
            dynamic_info += f'Best Value: {self.global_best_value:.6f}\n'
            ax_text_right.clear()
            ax_text_right.axis('off')
            ax_text_right.text(0.05, 0.5, dynamic_info, fontsize=10, va='center')

            self.history['positions'].append(positions.copy())
            self.history['best_positions'].append(self.global_best_position.copy())
            self.history['values'].append(self.global_best_value)
            self.history['times'].append(current_time)

            return scat, best_scat

        ani = animation.FuncAnimation(fig, update, frames=self.max_iter, init_func=init,
                                      blit=False, interval=200, repeat=False)
        plt.tight_layout()
        plt.show()

        # Animate best position over iterations
        self.animate_best_position()

    def animate_best_position(self):
        """
        Create an animation showing the movement of the global best position over iterations.
        """
        if self.dimensions == 2:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(np.ravel(X), np.ravel(Y))])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax)

            # **Add the constraint boundaries here**
            if self.constr_funcs:
                constraint_handles = []
                constraint_labels = []
                constraint_colors = ['white', 'yellow']  # Use different colors if you have more constraints
                for idx, constr_func in enumerate(self.constr_funcs):
                    # Evaluate the constraint function over the grid
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
                    constraint_values = constraint_values.reshape(X.shape)
                    # Plot the zero level contour of the constraint function
                    contour_set = ax.contour(X, Y, constraint_values, levels=[0], 
                                            colors=constraint_colors[idx % len(constraint_colors)], 
                                            linewidths=2, linestyles='dashed')
                    # Capture the contour lines for the legend
                    constraint_handles.append(contour_set.collections[0])
                    constraint_labels.append(f'Constraint {idx+1}')

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            best_positions = np.array(self.history['best_positions'])

            # Initialize scatter plot
            best_scat, = ax.plot([], [], 'bo-', label='Global Best Path')
            if self.analytic_solution is not None:
                ax.plot(self.analytic_solution[0], self.analytic_solution[1],
                        'gx', label='Analytic Solution', markersize=10, markeredgewidth=2)
            
            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            if self.constr_funcs:
                handles += constraint_handles
                labels += constraint_labels
            ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

            def init():
                best_scat.set_data([], [])
                return best_scat,

            def update(frame):
                xdata = best_positions[:frame+1, 0]
                ydata = best_positions[:frame+1, 1]
                best_scat.set_data(xdata, ydata)
                return best_scat,

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
        Plot the history of the best objective function value and best positions over iterations.
        """
        iterations = range(len(self.history['values']))

        # Plot Best Value over Iterations
        plt.figure()
        plt.plot(iterations, self.history['values'], marker='o')
        plt.title('Best Objective Function Value Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Value')
        plt.grid(True)
        plt.show()

        # Plot Best Position over Iterations (for each dimension)
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
    Gradient Descent optimization algorithm.
    """

    # Init method added
    def __init__(self, objective_func, bounds, max_iter, constr_funcs=None, analytic_solution=None, alpha=0.025): # Was 0.025
        super().__init__(objective_func, bounds, max_iter, constr_funcs, analytic_solution)
        self.alpha = alpha  # Step size

    def optimize(self):
        """
        Run the Gradient Descent algorithm and visualize the results.
        """
        start_time = time.time()

        # Starting point
        x_current = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        self.history['positions'].append(x_current.copy())
        self.history['values'].append(self.objective_func(x_current))
        self.history['times'].append(0)

        # Adjust figure layout
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 5, 1.2])

        # Left axis for static text
        ax_text_left = fig.add_subplot(gs[0])
        ax_text_left.axis('off')  # Hide the axis

        # Right axis for dynamic text
        ax_text_right = fig.add_subplot(gs[2])
        ax_text_right.axis('off')  # Hide the axis

        # Middle axis for the animation
        ax = fig.add_subplot(gs[1])
        ax.set_box_aspect(1)  # Make contour plot square

        # Adjust the subplot parameters to reduce left margin
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        if self.dimensions == 2:
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 200)
            y = np.linspace(y_min, y_max, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax)

            # Plot the constraint boundaries
            if self.constr_funcs:
                for idx, constr_func in enumerate(self.constr_funcs):
                    # Evaluate the constraint function over the grid
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
                    constraint_values = constraint_values.reshape(X.shape)
                    # Plot the zero level contour of the constraint function
                    ax.contour(X, Y, constraint_values, levels=[0], colors='white', linewidths=2, linestyles='dashed')

        # Prepare static information text
        static_info = 'Gradient Descent Optimization\n\n'

        # Objective function representation
        static_info += 'Objective Function:\n'
        static_info += TEX_STRING

        # Constraint functions representation
        if self.constr_funcs:
            static_info += 'Constraints:\n'
            for idx, constr_func in enumerate(self.constr_funcs, start=1):
                # For this example, we'll hardcode the constraint expressions
                if idx == 1:
                    static_info += r'$x + y - 1 \leq 0$' + '\n'
                elif idx == 2:
                    static_info += r'$x^2 + y^2 - 4 \leq 0$' + '\n'
            static_info += '\n'
        
        # Parameters
        static_info += 'Parameters:\n'
        static_info += f'Step Size (alpha): {self.alpha}\n\n'

        # Analytic solution
        if self.analytic_solution is not None:
            static_info += 'Analytic Solution:\n'
            static_info += f'{np.round(self.analytic_solution, 4)}\n'

        ax_text_left.text(0, 0.5, static_info, fontsize=10, va='center')

        point_scat = ax.scatter([], [], c='red', label='Current Point', s=50)
        if self.analytic_solution is not None:
            ax.scatter(self.analytic_solution[0], self.analytic_solution[1],
                       c='green', label='Analytic Solution', marker='x', s=100)
        
        # Adjust legend
        ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

        def init():
            # Initialize scatter plot with empty data
            empty_data = np.empty((0, 2))
            point_scat.set_offsets(empty_data)
            return point_scat,

        def gradient(x):
            """Compute the gradient of the objective function."""
            h = 1e-5  # Small step size for numerical gradient
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_step = x.copy()
                x_step[i] += h
                grad[i] = (self.objective_func(x_step) - self.objective_func(x)) / h
            return grad

        def project_onto_constraints(x):
            """Project x onto the feasible region defined by constraints."""
            from scipy.optimize import minimize

            def distance(y):
                return np.linalg.norm(y - x)

            cons = [{'type': 'ineq', 'fun': constr_func} for constr_func in self.constr_funcs]
            bounds = self.bounds.tolist()
            res = minimize(distance, x, bounds=bounds, constraints=cons)
            return res.x if res.success else x

        def update(frame):
            nonlocal x_current
            current_time = time.time() - start_time

            grad = gradient(x_current)
            x_next = x_current - self.alpha * grad

            # Apply constraints using projection
            if self.constr_funcs:
                x_next = project_onto_constraints(x_next)

            x_current = x_next
            value = self.objective_func(x_current)

            self.history['positions'].append(x_current.copy())
            self.history['values'].append(value)
            self.history['times'].append(current_time)

            point_scat.set_offsets([x_current])

            # Update dynamic text on the right
            iteration_number = frame + 1  # Adjust iteration count
            dynamic_info = f'Iteration: {iteration_number}\n'
            dynamic_info += f'Time Elapsed: {current_time:.2f}s\n'
            dynamic_info += f'Current Position: {np.round(x_current, 4)}\n'
            dynamic_info += f'Objective Value: {value:.6f}\n'
            ax_text_right.clear()
            ax_text_right.axis('off')
            ax_text_right.text(0.05, 0.5, dynamic_info, fontsize=10, va='center')

            return point_scat,

        ani = animation.FuncAnimation(fig, update, frames=self.max_iter, init_func=init,
                                      blit=False, interval=200, repeat=False)
        plt.tight_layout()
        plt.show()

        # Plot results
        self.plot_results('Gradient Descent')


class NewtonMethod(OptimizationAlgorithm):
    """
    Newton's Method optimization algorithm.
    """
    def optimize(self):
        """
        Run Newton's Method and visualize the results.
        """
        start_time = time.time()

        # Initial guess
        x_current = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        self.history['positions'].append(x_current.copy())
        self.history['values'].append(self.objective_func(x_current))
        self.history['times'].append(0)

        # Adjust figure layout
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 5, 1.2])

        # Left axis for static text
        ax_text_left = fig.add_subplot(gs[0])
        ax_text_left.axis('off')  # Hide the axis

        # Right axis for dynamic text
        ax_text_right = fig.add_subplot(gs[2])
        ax_text_right.axis('off')  # Hide the axis

        # Middle axis for the animation
        ax = fig.add_subplot(gs[1])
        ax.set_box_aspect(1)  # Make contour plot square

        # Adjust the subplot parameters to reduce left margin
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        if self.dimensions == 2:
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 200)
            y = np.linspace(y_min, y_max, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax)

            # Plot the constraint boundaries
            if self.constr_funcs:
                for idx, constr_func in enumerate(self.constr_funcs):
                    # Evaluate the constraint function over the grid
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
                    constraint_values = constraint_values.reshape(X.shape)
                    # Plot the zero level contour of the constraint function
                    ax.contour(X, Y, constraint_values, levels=[0], colors='white', linewidths=2, linestyles='dashed')

        # Prepare static information text
        static_info = "Newton's Method Optimization\n\n"

        # Objective function representation
        static_info += 'Objective Function:\n'
        static_info += TEX_STRING

        # Constraint functions representation
        if self.constr_funcs:
            static_info += 'Constraints:\n'
            for idx, constr_func in enumerate(self.constr_funcs, start=1):
                if idx == 1:
                    static_info += r'$x + y - 1 \leq 0$' + '\n'
                elif idx == 2:
                    static_info += r'$x^2 + y^2 - 4 \leq 0$' + '\n'
            static_info += '\n'

        # Analytic solution
        if self.analytic_solution is not None:
            static_info += 'Analytic Solution:\n'
            static_info += f'{np.round(self.analytic_solution, 4)}\n'

        ax_text_left.text(0, 0.5, static_info, fontsize=10, va='center')

        point_scat = ax.scatter([], [], c='red', label='Current Point', s=50)
        if self.analytic_solution is not None:
            ax.scatter(self.analytic_solution[0], self.analytic_solution[1],
                       c='green', label='Analytic Solution', marker='x', s=100)
        # Adjust legend
        ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

        # Prepare constraints in scipy format
        cons = []
        if self.constr_funcs:
            for constr_func in self.constr_funcs:
                cons.append({'type': 'ineq', 'fun': constr_func})

        # Bounds
        bounds = self.bounds.tolist()

        # Initialize iteration counter
        self.iteration = 0

        # Storage for positions and times
        self.positions = [x_current.copy()]
        self.times = [0]

        def callback(xk, state=None):
            self.iteration += 1
            current_time = time.time() - start_time
            value = self.objective_func(xk)

            self.history['positions'].append(xk.copy())
            self.history['values'].append(value)
            self.history['times'].append(current_time)
            self.positions.append(xk.copy())
            self.times.append(current_time)

        # Run optimization
        res = minimize(self.objective_func, x_current, method='trust-constr',
                       jac=objective_gradient, hess=objective_hessian,
                       constraints=cons if self.constr_funcs else None, bounds=bounds, callback=callback,
                       options={'maxiter': self.max_iter, 'verbose': 0})

        # Prepare animation data
        positions = np.array(self.history['positions'])

        def init():
            # Initialize scatter plot with empty data
            empty_data = np.empty((0, 2))
            point_scat.set_offsets(empty_data)
            return point_scat,

        def update(frame):
            if frame < len(positions):
                xk = positions[frame]
                time_elapsed = self.history['times'][frame]
                value = self.history['values'][frame]

                point_scat.set_offsets([xk])

                # Update dynamic text on the right
                iteration_number = frame + 1  # Adjust iteration count
                dynamic_info = f'Iteration: {iteration_number}\n'
                dynamic_info += f'Time Elapsed: {time_elapsed:.2f}s\n'
                dynamic_info += f'Current Position: {np.round(xk, 4)}\n'
                dynamic_info += f'Objective Value: {value:.6f}\n'
                ax_text_right.clear()
                ax_text_right.axis('off')
                ax_text_right.text(0.05, 0.5, dynamic_info, fontsize=10, va='center')

            return point_scat,

        ani = animation.FuncAnimation(fig, update, frames=len(positions), init_func=init,
                                      blit=False, interval=500, repeat=False)
        plt.tight_layout()
        plt.show()

        # Plot results
        self.plot_results("Newton's Method")


class BFGSMethod(OptimizationAlgorithm):
    """
    BFGS optimization algorithm.
    """
    def optimize(self):
        """
        Run BFGS method and visualize the results.
        """
        start_time = time.time()

        # Initial guess
        x_current = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        self.history['positions'].append(x_current.copy())
        self.history['values'].append(self.objective_func(x_current))
        self.history['times'].append(0)

        # Adjust figure layout
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 5, 1.2])

        # Left axis for static text
        ax_text_left = fig.add_subplot(gs[0])
        ax_text_left.axis('off')  # Hide the axis

        # Right axis for dynamic text
        ax_text_right = fig.add_subplot(gs[2])
        ax_text_right.axis('off')  # Hide the axis

        # Middle axis for the animation
        ax = fig.add_subplot(gs[1])
        ax.set_box_aspect(1)  # Make contour plot square

        # Adjust the subplot parameters to reduce left margin
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        if self.dimensions == 2:
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x = np.linspace(x_min, x_max, 200)
            y = np.linspace(y_min, y_max, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.array([self.objective_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
            Z = Z.reshape(X.shape)
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax)

            # Plot the constraint boundaries
            if self.constr_funcs:
                for idx, constr_func in enumerate(self.constr_funcs):
                    # Evaluate the constraint function over the grid
                    constraint_values = np.array([constr_func([x_val, y_val]) for x_val, y_val in zip(X.flatten(), Y.flatten())])
                    constraint_values = constraint_values.reshape(X.shape)
                    # Plot the zero level contour of the constraint function
                    ax.contour(X, Y, constraint_values, levels=[0], colors='white', linewidths=2, linestyles='dashed')

        # Prepare static information text
        static_info = 'BFGS Optimization\n\n'

        # Objective function representation
        static_info += 'Objective Function:\n'
        static_info += TEX_STRING

        # Constraint functions representation
        if self.constr_funcs:
            static_info += 'Constraints:\n'
            for idx, constr_func in enumerate(self.constr_funcs, start=1):
                if idx == 1:
                    static_info += r'$x + y - 1 \leq 0$' + '\n'
                elif idx == 2:
                    static_info += r'$x^2 + y^2 - 4 \leq 0$' + '\n'
            static_info += '\n'

        # Analytic solution
        if self.analytic_solution is not None:
            static_info += 'Analytic Solution:\n'
            static_info += f'{np.round(self.analytic_solution, 4)}\n'

        ax_text_left.text(0, 0.5, static_info, fontsize=10, va='center')

        point_scat = ax.scatter([], [], c='red', label='Current Point', s=50)
        if self.analytic_solution is not None:
            ax.scatter(self.analytic_solution[0], self.analytic_solution[1],
                       c='green', label='Analytic Solution', marker='x', s=100)
        # Adjust legend
        ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)

        # Prepare constraints in scipy format
        cons = []
        if self.constr_funcs:
            for constr_func in self.constr_funcs:
                cons.append({'type': 'ineq', 'fun': constr_func})

        # Bounds
        bounds = self.bounds.tolist()

        # Initialize iteration counter
        self.iteration = 0

        # Storage for positions and times
        self.positions = [x_current.copy()]
        self.times = [0]

        # Define callback function
        def callback(xk):
            self.iteration += 1
            current_time = time.time() - start_time
            value = self.objective_func(xk)

            self.history['positions'].append(xk.copy())
            self.history['values'].append(value)
            self.history['times'].append(current_time)
            self.positions.append(xk.copy())
            self.times.append(current_time)

        method = 'SLSQP' if self.constr_funcs else 'L-BFGS-B'
        res = minimize(self.objective_func, x_current, method=method,
                    jac=objective_gradient,
                    constraints=cons if self.constr_funcs else None, bounds=bounds, callback=callback,
                    options={'maxiter': self.max_iter, 'disp': False})

        # Prepare animation data
        positions = np.array(self.history['positions'])

        def init():
            # Initialize scatter plot with empty data
            empty_data = np.empty((0, 2))
            point_scat.set_offsets(empty_data)
            return point_scat,

        def update(frame):
            if frame < len(positions):
                xk = positions[frame]
                time_elapsed = self.history['times'][frame]
                value = self.history['values'][frame]

                point_scat.set_offsets([xk])

                # Update dynamic text on the right
                iteration_number = frame + 1  # Adjust iteration count
                dynamic_info = f'Iteration: {iteration_number}\n'
                dynamic_info += f'Time Elapsed: {time_elapsed:.2f}s\n'
                dynamic_info += f'Current Position: {np.round(xk, 4)}\n'
                dynamic_info += f'Objective Value: {value:.6f}\n'
                ax_text_right.clear()
                ax_text_right.axis('off')
                ax_text_right.text(0.05, 0.5, dynamic_info, fontsize=10, va='center')

            return point_scat,

        ani = animation.FuncAnimation(fig, update, frames=len(positions), init_func=init,
                                      blit=False, interval=500, repeat=False)
        plt.tight_layout()
        plt.show()

        # Plot results
        self.plot_results('BFGS Method')


def run_comparison_mode(objective_func, bounds, max_iter, constr_funcs, analytic_solution):
    """
    Run the comparison mode to compare different optimization algorithms.
    """
    # Instantiate algorithms
    pso = PSO(objective_func, bounds, num_particles=30, max_iter=max_iter,
              w=0.5, c1=1.5, c2=1.5, constr_funcs=constr_funcs, analytic_solution=analytic_solution)

    gd = GradientDescent(objective_func, bounds, max_iter, constr_funcs, analytic_solution)

    newton = NewtonMethod(objective_func, bounds, max_iter, constr_funcs, analytic_solution)

    bfgs = BFGSMethod(objective_func, bounds, max_iter, constr_funcs, analytic_solution)

    # Run optimizations
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

    # Define bounds for each dimension: [(min, max), (min, max), ...]
    bounds = [(-10, 10), (-10, 10)]  # [(-5.12, 5.12), (-5.12, 5.12)] initially

    # Ask user for analytic solution
    analytic_solution_input = input("Enter the analytic solution coordinates separated by commas (or press Enter to skip): ")
    if analytic_solution_input.strip():
        analytic_solution = np.array([float(coord) for coord in analytic_solution_input.split(',')])
    else:
        analytic_solution = None

    # Ask user if they want to run in PSO mode or comparison mode
    mode_input = input("Enter '1' to run PSO mode or '2' for comparison mode: ")

    # Ask user if they want to solve constrained or unconstrained problem
    constraint_input = input("Enter '1' for constrained optimization or '2' for unconstrained optimization: ")
    if constraint_input.strip() == '1':
        constr_funcs = constraint_functions
    elif constraint_input.strip() == '2':
        constr_funcs = None
    else:
        print("Invalid input. Exiting.")
        exit()

    if mode_input.strip() == '1':
        # Initialize PSO with constraints and analytic solution
        pso = PSO(objective_function, bounds, num_particles=30, max_iter=50,
                  w=0.5, c1=1.5, c2=1.5, constr_funcs=constr_funcs, analytic_solution=analytic_solution) 

        # Run optimization
        pso.optimize()
    elif mode_input.strip() == '2':
        # Run comparison mode
        run_comparison_mode(objective_function, bounds, max_iter=50, constr_funcs=constr_funcs, analytic_solution=analytic_solution)
    else:
        print("Invalid input. Exiting.")

# w = 0.5, c1 = 1.5, c2 = 1.5
