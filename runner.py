import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fem1d import FEM1D
from fem2d import FEM2D
import time

def main():
    print("FEM Solver for Poisson Equation")
    print("================================")
    
    # Solve 1D Poisson problem
    print("\n1D Poisson Problem")
    print("-----------------")
    
    # Define problem parameters
    L = 1.0  # Domain length [0,L]
    n_elements = [10, 20, 40, 80]  # Number of elements to test
    
    # Source function f(x) = 2 for the equation -u''(x) = f(x)
    # The second derivative of x(1-x) is -2, so for -u''(x) = f(x), we need f(x) = 2
    f_1d = lambda x: 2.0
    
    # Analytical solution u(x) = x(1-x)
    analytical_1d = lambda x: x * (1 - x)
    
    # Store errors for convergence analysis
    errors_1d = []
    
    plt.figure(figsize=(12, 10))
    
    for i, n in enumerate(n_elements):
        # Create solver instance
        solver = FEM1D(0, L, n, f_1d)
        
        # Solve the problem
        start_time = time.time()
        solver.solve()
        end_time = time.time()
        
        # Get solution
        x_fem = solver.mesh
        u_fem = solver.solution
        
        # Calculate analytical solution
        x_analytical = np.linspace(0, L, 100)
        u_analytical = analytical_1d(x_analytical)
        
        # Calculate error
        u_analytical_at_nodes = analytical_1d(x_fem)
        error = np.sqrt(np.sum((u_fem - u_analytical_at_nodes)**2) / len(u_fem))
        errors_1d.append(error)
        
        # Plot the solutions
        plt.subplot(2, 2, i + 1)
        plt.plot(x_analytical, u_analytical, 'r-', label='Analytical')
        plt.plot(x_fem, u_fem, 'bo-', label=f'FEM (n={n})')
        plt.legend()
        plt.title(f'1D FEM, n={n}, Error={error:.2e}, Time={end_time-start_time:.4f}s')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fem1d_comparison.png')
    
    # Plot convergence
    plt.figure(figsize=(8, 6))
    plt.loglog(n_elements, errors_1d, 'bo-')
    plt.xlabel('Number of Elements')
    plt.ylabel('Error (L2 norm)')
    plt.title('1D FEM Error Convergence')
    plt.grid(True)
    plt.savefig('fem1d_convergence.png')
    
    print("\n2D Poisson Problem")
    print("-----------------")
    
    # Define problem parameters for 2D
    nx_values = [10, 20]  # Number of elements in x direction
    ny_values = [10, 20]  # Number of elements in y direction
    
    # For analytical solution u(x,y) = x(1-x)y(1-y):
    # Computing -∆u = -[u_xx + u_yy]:
    # u_xx = -2y(1-y)
    # u_yy = -2x(1-x)
    # -∆u = 2y(1-y) + 2x(1-x) = 2[y(1-y) + x(1-x)]
    f_2d = lambda x, y: 2 * (y*(1-y) + x*(1-x))
    
    # Analytical solution u(x,y) = x*(1-x)*y*(1-y)
    analytical_2d = lambda x, y: x * (1 - x) * y * (1 - y)
    
    # Store errors for convergence analysis
    errors_2d = []
    
    for i, (nx, ny) in enumerate(zip(nx_values, ny_values)):
        # Create solver instance for the unit square [0,1]×[0,1]
        solver = FEM2D(0, 1, 0, 1, nx, ny, f_2d)
        
        # Solve the problem
        start_time = time.time()
        solver.solve()
        end_time = time.time()
        
        # Get solution
        nodes = solver.nodes
        triangles = solver.triangles
        solution = solver.solution
        
        # Calculate error at nodes
        analytical_values = np.array([analytical_2d(node[0], node[1]) for node in nodes])
        error = np.sqrt(np.sum((solution - analytical_values)**2) / len(solution))
        errors_2d.append(error)
        
        print(f"2D FEM with {nx}x{ny} elements:")
        print(f"  - Error: {error:.2e}")
        print(f"  - Solve time: {end_time-start_time:.4f} seconds")
        
        # Create a regular grid for plotting analytical solution
        x_grid = np.linspace(0, 1, 50)
        y_grid = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z_analytical = analytical_2d(X, Y)
        
        # Plot solutions
        fig = plt.figure(figsize=(14, 6))
        
        # Plot FEM solution
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_trisurf(nodes[:, 0], nodes[:, 1], solution, triangles=triangles, cmap='viridis', edgecolor='none')
        ax1.set_title(f'FEM Solution (nx={nx}, ny={ny})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u(x,y)')
        
        # Plot analytical solution
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, Z_analytical, cmap='viridis', edgecolor='none')
        ax2.set_title('Analytical Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u(x,y)')
        
        plt.tight_layout()
        plt.savefig(f'fem2d_comparison_{nx}x{ny}.png')
    
    # Plot convergence
    plt.figure(figsize=(8, 6))
    plt.loglog([n*n for n in nx_values], errors_2d, 'bo-')
    plt.xlabel('Number of Elements (Total)')
    plt.ylabel('Error (L2 norm)')
    plt.title('2D FEM Error Convergence')
    plt.grid(True)
    plt.savefig('fem2d_convergence.png')
    
    # Define a more complex analytical solution
    # u(x,y) = sin(π*x)*sin(2π*y) + 0.5*sin(3π*x)*sin(4π*y)
    analytical_2d_complex = lambda x, y: np.sin(np.pi*x)*np.sin(2*np.pi*y) + 0.5*np.sin(3*np.pi*x)*np.sin(4*np.pi*y)
    
    # Computing the corresponding source term f(x,y) = -∆u = -[u_xx + u_yy]
    # u_xx = -π²*sin(π*x)*sin(2π*y) - 4.5π²*sin(3π*x)*sin(4π*y)
    # u_yy = -4π²*sin(π*x)*sin(2π*y) - 8π²*0.5*sin(3π*x)*sin(4π*y)
    # -∆u = (5π²*sin(π*x)*sin(2π*y) + 12.5π²*sin(3π*x)*sin(4π*y))
    f_2d_complex = lambda x, y: (5*np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y) + 
                                12.5*np.pi**2*np.sin(3*np.pi*x)*np.sin(4*np.pi*y))
    
    print("\n2D Poisson Problem with Complex Solution")
    print("--------------------------------------")
    
    # Higher resolution for complex solution
    nx_values_complex = [20, 40]
    ny_values_complex = [20, 40]
    errors_2d_complex = []
    
    for i, (nx, ny) in enumerate(zip(nx_values_complex, ny_values_complex)):
        # Create solver instance for the unit square [0,1]×[0,1]
        solver = FEM2D(0, 1, 0, 1, nx, ny, f_2d_complex)
        
        # Solve the problem
        start_time = time.time()
        solver.solve()
        end_time = time.time()
        
        # Get solution
        nodes = solver.nodes
        triangles = solver.triangles
        solution = solver.solution
        
        # Calculate error at nodes
        analytical_values = np.array([analytical_2d_complex(node[0], node[1]) for node in nodes])
        error = np.sqrt(np.sum((solution - analytical_values)**2) / len(solution))
        errors_2d_complex.append(error)
        
        print(f"Complex 2D FEM with {nx}x{ny} elements:")
        print(f"  - Error: {error:.2e}")
        print(f"  - Solve time: {end_time-start_time:.4f} seconds")
        
        # Create a regular grid for plotting analytical solution
        x_grid = np.linspace(0, 1, 100)  # Higher resolution for complex function
        y_grid = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z_analytical = analytical_2d_complex(X, Y)
        
        # Plot solutions
        fig = plt.figure(figsize=(14, 6))
        
        # Plot FEM solution
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_trisurf(nodes[:, 0], nodes[:, 1], solution, triangles=triangles, cmap='viridis', edgecolor='none')
        ax1.set_title(f'FEM Solution (nx={nx}, ny={ny})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u(x,y)')
        
        # Plot analytical solution
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, Z_analytical, cmap='viridis', edgecolor='none')
        ax2.set_title('Complex Analytical Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u(x,y)')
        
        plt.tight_layout()
        plt.savefig(f'fem2d_complex_comparison_{nx}x{ny}.png')
    
    # Plot convergence for complex solution
    plt.figure(figsize=(8, 6))
    plt.loglog([n*n for n in nx_values_complex], errors_2d_complex, 'ro-')
    plt.xlabel('Number of Elements (Total)')
    plt.ylabel('Error (L2 norm)')
    plt.title('2D FEM Error Convergence for Complex Solution')
    plt.grid(True)
    plt.savefig('fem2d_complex_convergence.png')
    
    print("\nAll computations completed. Results saved as PNG files.")

if __name__ == "__main__":
    main()