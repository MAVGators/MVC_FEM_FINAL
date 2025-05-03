# Finite Element Method (FEM) Solver for Poisson Equation

This project implements numerical solvers for the Poisson equation in both 1D and 2D domains using the Finite Element Method (FEM). It's designed to be educational and demonstrate the core concepts of FEM applied to a classic partial differential equation problem.

## Mathematical Background

### The Poisson Equation

The Poisson equation is a second-order elliptic partial differential equation:

**1D**: $-u''(x) = f(x)$ on the interval $[a,b]$

**2D**: $-\Delta u(x,y) = f(x,y)$ on the domain $\Omega \subset \mathbb{R}^2$

where:
- $u$ is the unknown solution function
- $f$ is a known source function
- $\Delta$ is the Laplace operator ($\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$)

In both cases, we apply Dirichlet boundary conditions where $u = 0$ on the boundary.

### Finite Element Method (FEM)

FEM solves the PDE by:
1. Discretizing the domain into elements (line segments in 1D, triangles in 2D)
2. Constructing a weak form of the differential equation
3. Approximating the solution with piecewise polynomial basis functions
4. Assembling a linear system $A\mathbf{u} = \mathbf{f}$
5. Solving the linear system to find the approximate solution at node points

## Project Structure

- `fem1d.py`: Implementation of the 1D FEM solver
- `fem2d.py`: Implementation of the 2D FEM solver with triangular elements
- `runner.py`: Main script to run both solvers and compare with analytical solutions

## Features

- 1D FEM solver with linear elements
- 2D FEM solver with triangular elements
- Comparison with analytical solutions
- Error analysis and convergence study
- Visualization of numerical and analytical solutions

## Example Problems

### 1D Problem
- Equation: $-u''(x) = -2$ on $[0,1]$
- Boundary conditions: $u(0) = u(1) = 0$
- Analytical solution: $u(x) = x(1-x)$

### 2D Problem
- Equation: $-\Delta u(x,y) = -2(x^2 + y^2)$ on $[0,1] \times [0,1]$
- Boundary conditions: $u = 0$ on the boundary
- Analytical solution: $u(x,y) = x(1-x)y(1-y)$

## Usage

To run the code:

```bash
python runner.py
```

This will execute both 1D and 2D solvers with different mesh resolutions and generate plots comparing the numerical solutions with the analytical ones.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Output

The code generates several output files:
- `fem1d_comparison.png`: Comparison of 1D FEM solutions with analytical solution
- `fem1d_convergence.png`: Convergence plot for the 1D solver
- `fem2d_comparison_10x10.png` and `fem2d_comparison_20x20.png`: Comparison of 2D FEM solutions with analytical solution
- `fem2d_convergence.png`: Convergence plot for the 2D solver

## Theory Notes

### Basis Functions

In 1D, we use piecewise linear basis functions (hat functions) defined for each node. In 2D, we use linear triangular elements with basis functions that are linear over each triangle.

### Weak Form

The weak form of the Poisson equation is derived by multiplying by a test function $v$ and integrating by parts:

$$\int_{\Omega} \nabla u \cdot \nabla v \, dx = \int_{\Omega} f v \, dx$$

### Linear System

The resulting linear system has:
- Stiffness matrix $A_{ij} = \int_{\Omega} \nabla \phi_i \cdot \nabla \phi_j \, dx$
- Load vector $F_i = \int_{\Omega} f \phi_i \, dx$

where $\phi_i$ are the basis functions.