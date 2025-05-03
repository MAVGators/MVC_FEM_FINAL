import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

class FEM1D:
    """
    Finite Element Method solver for the 1D Poisson equation:
    -u''(x) = f(x) on [a,b]
    with Dirichlet boundary conditions u(a) = u(b) = 0.
    """
    
    def __init__(self, a, b, n_elements, f):
        """
        Initialize the 1D FEM solver.
        
        Parameters:
            a (float): Left boundary of the domain
            b (float): Right boundary of the domain
            n_elements (int): Number of elements
            f (function): Source function f(x) in -u''(x) = f(x)
        """
        self.a = a
        self.b = b
        self.n_elements = n_elements
        self.f = f
        
        # Create the mesh (n_elements + 1 nodes)
        self.mesh = np.linspace(a, b, n_elements + 1)
        self.h = (b - a) / n_elements  # Element size
        
        # Initialize solution
        self.solution = None
    
    def assemble_system(self):
        """
        Assemble the FEM system matrices and right-hand side vector.
        """
        n_nodes = self.n_elements + 1
        
        # Initialize sparse stiffness matrix and load vector
        A = lil_matrix((n_nodes, n_nodes))
        F = np.zeros(n_nodes)
        
        # Assemble the stiffness matrix and load vector element-wise
        for e in range(self.n_elements):
            x_left = self.mesh[e]
            x_right = self.mesh[e + 1]
            
            # Element stiffness matrix: [1 -1; -1 1] / h
            A[e, e] += 1.0 / self.h
            A[e, e + 1] += -1.0 / self.h
            A[e + 1, e] += -1.0 / self.h
            A[e + 1, e + 1] += 1.0 / self.h
            
            # Element load vector: [f*h/2; f*h/2] (trapezoidal rule)
            # For linear finite elements, we integrate the product of f and the basis functions
            f_left = self.f(x_left)
            f_right = self.f(x_right)
            
            F[e] += (f_left * 2 + f_right) * self.h / 6.0
            F[e + 1] += (f_left + f_right * 2) * self.h / 6.0
        
        # Convert to CSR format for efficient solving
        self.A = A.tocsr()
        self.F = F
    
    def apply_boundary_conditions(self):
        """
        Apply Dirichlet boundary conditions u(a) = u(b) = 0.
        """
        n_nodes = self.n_elements + 1
        
        # Set the first and last rows to identity rows (boundary nodes)
        self.A[0, :] = 0.0
        self.A[0, 0] = 1.0
        self.F[0] = 0.0
        
        self.A[n_nodes - 1, :] = 0.0
        self.A[n_nodes - 1, n_nodes - 1] = 1.0
        self.F[n_nodes - 1] = 0.0
    
    def solve(self):
        """
        Solve the FEM system and compute the solution.
        """
        # Assemble system
        self.assemble_system()
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
        
        # Solve the system
        self.solution = spsolve(self.A, self.F)
        
        return self.solution
    
    def evaluate_at(self, x_points):
        """
        Evaluate the solution at arbitrary points using linear interpolation.
        
        Parameters:
            x_points (array): Points at which to evaluate the solution
            
        Returns:
            array: Solution values at the specified points
        """
        if self.solution is None:
            raise ValueError("Solution not computed. Call solve() first.")
        
        # Initialize result array
        u_interp = np.zeros_like(x_points, dtype=float)
        
        # For each point, find its element and interpolate
        for i, x in enumerate(x_points):
            # Skip points outside the domain
            if x < self.a or x > self.b:
                continue
                
            # Find element containing x
            if x == self.b:
                e = self.n_elements - 1
            else:
                e = int((x - self.a) / self.h)
            
            # Local coordinate within element [0, 1]
            xi = (x - self.mesh[e]) / self.h
            
            # Linear interpolation
            u_interp[i] = (1 - xi) * self.solution[e] + xi * self.solution[e + 1]
        
        return u_interp