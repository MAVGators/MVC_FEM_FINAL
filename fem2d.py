import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

class FEM2D:
    """
    Finite Element Method solver for the 2D Poisson equation:
    -∆u(x,y) = f(x,y) on [a,b] × [c,d]
    with Dirichlet boundary conditions u = 0 on the boundary.
    """
    
    def __init__(self, a, b, c, d, nx, ny, f):
        """
        Initialize the 2D FEM solver.
        
        Parameters:
            a, b (float): Domain boundaries in x direction [a,b]
            c, d (float): Domain boundaries in y direction [c,d]
            nx, ny (int): Number of elements in x and y directions
            f (function): Source function f(x,y) in -∆u = f
        """
        self.a, self.b = a, b
        self.c, self.d = c, d
        self.nx, self.ny = nx, ny
        self.f = f
        
        # Create the mesh
        self._create_mesh()
        
        # Initialize solution
        self.solution = None
    
    def _create_mesh(self):
        """
        Create a structured triangular mesh on the rectangular domain.
        """
        # Create a structured grid of points
        x = np.linspace(self.a, self.b, self.nx + 1)
        y = np.linspace(self.c, self.d, self.ny + 1)
        X, Y = np.meshgrid(x, y)
        
        # Reshape to get node coordinates
        self.nodes = np.vstack([X.flatten(), Y.flatten()]).T
        self.n_nodes = len(self.nodes)
        
        # Create triangulation
        self.triangles = []
        for j in range(self.ny):
            for i in range(self.nx):
                # Node indices for the four corners of this grid cell
                n00 = j * (self.nx + 1) + i
                n10 = n00 + 1
                n01 = (j + 1) * (self.nx + 1) + i
                n11 = n01 + 1
                
                # Add two triangles per grid cell
                self.triangles.append([n00, n10, n01])  # Lower-left triangle
                self.triangles.append([n10, n11, n01])  # Upper-right triangle
        
        self.triangles = np.array(self.triangles, dtype=int)
        self.n_triangles = len(self.triangles)
        
        # Identify boundary nodes (for Dirichlet BCs)
        self.boundary_nodes = []
        for i, node in enumerate(self.nodes):
            x, y = node
            if (abs(x - self.a) < 1e-10 or abs(x - self.b) < 1e-10 or 
                abs(y - self.c) < 1e-10 or abs(y - self.d) < 1e-10):
                self.boundary_nodes.append(i)
    
    def assemble_system(self):
        """
        Assemble the FEM system matrices and right-hand side vector.
        """
        # Initialize sparse stiffness matrix and load vector
        A = lil_matrix((self.n_nodes, self.n_nodes))
        F = np.zeros(self.n_nodes)
        
        # For each triangle element
        for t in range(self.n_triangles):
            # Get the triangle vertices
            vertex_indices = self.triangles[t]
            vertices = self.nodes[vertex_indices]
            
            # Compute triangle area
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]
            
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            
            # Compute gradient of basis functions
            # For linear elements on triangles, the gradient is constant
            b = np.array([y2 - y3, y3 - y1, y1 - y2])
            c = np.array([x3 - x2, x1 - x3, x2 - x1])
            
            # Scale by 1/(2*Area)
            b = b / (2 * area)
            c = c / (2 * area)
            
            # Compute local stiffness matrix
            for i in range(3):
                for j in range(3):
                    # Contribution to stiffness matrix: integral of grad(phi_i) · grad(phi_j)
                    A[vertex_indices[i], vertex_indices[j]] += area * (b[i] * b[j] + c[i] * c[j])
            
            # Compute local load vector using barycenter quadrature
            # For linear elements, we can approximate the integral using the centroid value
            centroid_x = (x1 + x2 + x3) / 3
            centroid_y = (y1 + y2 + y3) / 3
            f_centroid = self.f(centroid_x, centroid_y)
            
            for i in range(3):
                F[vertex_indices[i]] += area * f_centroid / 3
        
        # Convert to CSR format for efficient solving
        self.A = A.tocsr()
        self.F = F
    
    def apply_boundary_conditions(self):
        """
        Apply homogeneous Dirichlet boundary conditions.
        """
        # Set rows corresponding to boundary nodes to identity rows
        for node in self.boundary_nodes:
            self.A[node, :] = 0.0
            self.A[node, node] = 1.0
            self.F[node] = 0.0
    
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
    
    def plot_solution(self, ax=None, title='FEM Solution'):
        """
        Plot the solution using triangulation.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            title (str): Plot title
            
        Returns:
            matplotlib.axes.Axes: The axes object with the plot
        """
        if self.solution is None:
            raise ValueError("Solution not computed. Call solve() first.")
        
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Create triangulation object
        triang = mtri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.triangles)
        
        # Plot the surface
        surf = ax.plot_trisurf(triang, self.solution, cmap='viridis', edgecolor='none')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y)')
        ax.set_title(title)
        
        return ax