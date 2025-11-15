/*
 * Partial Differential Equation (PDE) Solver
 * Supports both ODEs and PDEs
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef PDE_SOLVER_H
#define PDE_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * PDE type enumeration
 */
typedef enum {
    PDE_HEAT,           // Heat/diffusion equation: u_t = α∇²u
    PDE_WAVE,           // Wave equation: u_tt = c²∇²u
    PDE_LAPLACE,        // Laplace equation: ∇²u = 0
    PDE_POISSON,        // Poisson equation: ∇²u = f(x,y)
    PDE_BURGERS,        // Burgers equation: u_t + uu_x = νu_xx
    PDE_ADVECTION       // Advection equation: u_t + au_x = 0
} PDEType;

/**
 * Spatial dimension
 */
typedef enum {
    DIM_1D = 1,
    DIM_2D = 2,
    DIM_3D = 3
} SpatialDimension;

/**
 * PDE problem structure
 */
typedef struct {
    PDEType type;
    SpatialDimension dim;
    double* initial_condition;  // Initial condition u(x,0) or u(x,y,0)
    double* boundary_condition; // Boundary conditions
    size_t nx, ny, nz;          // Grid points in each dimension
    double dx, dy, dz;          // Spatial step sizes
    double dt;                  // Time step
    double alpha;                // Diffusion coefficient (for heat equation)
    double c;                   // Wave speed (for wave equation)
    double nu;                  // Viscosity (for Burgers equation)
    double a;                   // Advection speed (for advection equation)
} PDEProblem;

/**
 * PDE solution structure
 */
typedef struct {
    double* u;                  // Solution array
    double* u_prev;             // Previous time step
    size_t n_points;           // Total number of grid points
    size_t n_time_steps;        // Number of time steps
    double* time;               // Time array
    double current_time;       // Current simulation time
} PDESolution;

/**
 * Initialize PDE problem
 * 
 * @param problem: PDE problem structure
 * @param type: Type of PDE
 * @param dim: Spatial dimension
 * @param nx, ny, nz: Grid points (ny, nz can be 1 for 1D)
 * @param dx, dy, dz: Spatial step sizes
 * @param dt: Time step
 * @return: 0 on success, -1 on failure
 */
int pde_problem_init(PDEProblem* problem, PDEType type, SpatialDimension dim,
                     size_t nx, size_t ny, size_t nz,
                     double dx, double dy, double dz, double dt);

/**
 * Free PDE problem resources
 */
void pde_problem_free(PDEProblem* problem);

/**
 * Solve 1D heat equation using finite difference method
 * u_t = α * u_xx
 * 
 * @param problem: PDE problem
 * @param t_end: Final time
 * @param solution: Output solution structure
 * @return: 0 on success, -1 on failure
 */
int pde_solve_heat_1d(const PDEProblem* problem, double t_end, PDESolution* solution);

/**
 * Solve 2D heat equation
 * u_t = α * (u_xx + u_yy)
 * 
 * @param problem: PDE problem
 * @param t_end: Final time
 * @param solution: Output solution structure
 * @return: 0 on success, -1 on failure
 */
int pde_solve_heat_2d(const PDEProblem* problem, double t_end, PDESolution* solution);

/**
 * Solve 1D wave equation
 * u_tt = c² * u_xx
 * 
 * @param problem: PDE problem
 * @param t_end: Final time
 * @param solution: Output solution structure
 * @return: 0 on success, -1 on failure
 */
int pde_solve_wave_1d(const PDEProblem* problem, double t_end, PDESolution* solution);

/**
 * Solve 1D advection equation
 * u_t + a * u_x = 0
 * 
 * @param problem: PDE problem
 * @param t_end: Final time
 * @param solution: Output solution structure
 * @return: 0 on success, -1 on failure
 */
int pde_solve_advection_1d(const PDEProblem* problem, double t_end, PDESolution* solution);

/**
 * Solve 1D Burgers equation
 * u_t + u * u_x = ν * u_xx
 * 
 * @param problem: PDE problem
 * @param t_end: Final time
 * @param solution: Output solution structure
 * @return: 0 on success, -1 on failure
 */
int pde_solve_burgers_1d(const PDEProblem* problem, double t_end, PDESolution* solution);

/**
 * Solve 2D Laplace equation
 * u_xx + u_yy = 0
 * 
 * @param problem: PDE problem
 * @param solution: Output solution structure
 * @return: 0 on success, -1 on failure
 */
int pde_solve_laplace_2d(const PDEProblem* problem, PDESolution* solution);

/**
 * Solve 2D Poisson equation
 * u_xx + u_yy = f(x,y)
 * 
 * @param problem: PDE problem
 * @param f: Source function f(x,y)
 * @param solution: Output solution structure
 * @return: 0 on success, -1 on failure
 */
int pde_solve_poisson_2d(const PDEProblem* problem, 
                        double (*f)(double x, double y, void* params),
                        void* params, PDESolution* solution);

/**
 * Free PDE solution resources
 */
void pde_solution_free(PDESolution* solution);

/**
 * Export solution to file
 * 
 * @param solution: PDE solution
 * @param problem: PDE problem
 * @param filename: Output filename
 * @return: 0 on success, -1 on failure
 */
int pde_export_solution(const PDESolution* solution, const PDEProblem* problem,
                       const char* filename);

#ifdef __cplusplus
}
#endif

#endif /* PDE_SOLVER_H */
