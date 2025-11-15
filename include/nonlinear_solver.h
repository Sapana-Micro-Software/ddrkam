/*
 * Nonlinear Programming-Based Solvers for ODEs and PDEs
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef NONLINEAR_SOLVER_H
#define NONLINEAR_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * PDE function pointer type
 */
typedef void (*PDEFunction)(double t, double x, double y, const double* u, 
                           double* dudt, double* dudx, double* dudy, void* params);

/**
 * Objective function for nonlinear programming
 */
typedef double (*ObjectiveFunction)(const double* x, size_t n, void* params);

/**
 * Constraint function for nonlinear programming
 */
typedef void (*ConstraintFunction)(const double* x, size_t n, double* constraints, void* params);

/**
 * Nonlinear Programming Solver Type
 */
typedef enum {
    NLP_GRADIENT_DESCENT,    // Gradient descent
    NLP_NEWTON,              // Newton's method
    NLP_QUASI_NEWTON,         // Quasi-Newton (BFGS)
    NLP_INTERIOR_POINT,       // Interior point method
    NLP_SEQUENTIAL_QP,        // Sequential quadratic programming
    NLP_TRUST_REGION          // Trust region method
} NLPSolverType;

/**
 * Interior Point Method Parameters
 */
typedef struct {
    double barrier_parameter;      // Initial barrier parameter (mu)
    double barrier_reduction;       // Barrier reduction factor (tau)
    double centering_parameter;     // Centering parameter (sigma)
    double feasibility_tolerance;   // Feasibility tolerance
    double optimality_tolerance;    // Optimality tolerance
    size_t max_barrier_iterations;  // Max iterations per barrier step
    int handle_nonconvex;           // Flag for non-convex handling
    double perturbation_radius;     // Perturbation for non-convex escape
} InteriorPointParams;

/**
 * Nonlinear ODE Solver
 * Uses nonlinear programming to solve ODEs as optimization problems
 */
typedef struct {
    size_t state_dim;
    NLPSolverType solver_type;
    
    // Optimization parameters
    double tolerance;
    size_t max_iterations;
    double step_size;
    
    // Gradient/hessian computation
    double* gradient;
    double** hessian;
    
    // Objective and constraints
    ObjectiveFunction objective;
    ConstraintFunction constraints;
    void* params;
    
    // History for quasi-Newton
    double* gradient_history;
    double* step_history;
    size_t history_size;
    
    // Interior Point Method specific
    InteriorPointParams ip_params;
    double* slack_variables;        // Slack variables for inequality constraints
    double* dual_variables;         // Dual (Lagrange multiplier) variables
    double* barrier_gradient;       // Gradient with barrier term
    double** barrier_hessian;      // Hessian with barrier term
    size_t num_constraints;         // Number of constraints
} NonlinearODESolver;

/**
 * Nonlinear PDE Solver
 * Uses nonlinear programming for PDE optimization
 */
typedef struct {
    size_t spatial_dim;      // Spatial dimensions (1D, 2D, 3D)
    size_t* grid_size;        // Grid size per dimension
    NLPSolverType solver_type;
    
    // Optimization parameters
    double tolerance;
    size_t max_iterations;
    
    // Grid data
    double* u_grid;          // Solution grid
    double* u_prev;          // Previous iteration
    double* residual;        // Residual vector
    
    // PDE function
    PDEFunction pde_func;
    void* params;
} NonlinearPDESolver;

/**
 * Distributed Nonlinear Solver
 * Combines nonlinear programming with distributed computing
 */
typedef struct {
    NonlinearODESolver* base_solver;
    size_t num_workers;
    int rank;                // MPI rank
    int size;                // MPI size
    void* mpi_comm;
    
    // Work distribution
    size_t* work_ranges;
    double* local_state;
    double* global_state;
} DistributedNonlinearSolver;

/**
 * Data-Driven Nonlinear Solver
 * Combines nonlinear programming with hierarchical learning
 */
typedef struct {
    NonlinearODESolver* base_solver;
    size_t num_layers;
    double** layer_weights;
    double* attention_weights;
    double learning_rate;
} DataDrivenNonlinearSolver;

/**
 * Online Nonlinear Solver
 * Adaptive nonlinear programming with incremental updates
 */
typedef struct {
    NonlinearODESolver* base_solver;
    double* adaptive_tolerance;
    double* adaptive_step_size;
    double learning_rate;
    size_t adaptation_count;
} OnlineNonlinearSolver;

/**
 * Real-Time Nonlinear Solver
 * Nonlinear programming optimized for streaming data
 */
typedef struct {
    NonlinearODESolver* base_solver;
    double* buffer;
    size_t buffer_size;
    void (*callback)(double t, const double* y, size_t n, void* data);
    void* callback_data;
} RealtimeNonlinearSolver;

// Nonlinear ODE Solver Functions
int nonlinear_ode_init(NonlinearODESolver* solver, size_t state_dim,
                       NLPSolverType solver_type, ObjectiveFunction objective,
                       ConstraintFunction constraints, void* params);
void nonlinear_ode_free(NonlinearODESolver* solver);
int nonlinear_ode_solve(NonlinearODESolver* solver, ODEFunction f,
                       double t0, double t_end, const double* y0,
                       double* y_out);

// Interior Point Method Configuration
int nonlinear_ode_set_constraints(NonlinearODESolver* solver, size_t num_constraints);
int nonlinear_ode_set_interior_point_params(NonlinearODESolver* solver,
                                           const InteriorPointParams* params);
int nonlinear_ode_enable_nonconvex(NonlinearODESolver* solver, int enable,
                                   double perturbation_radius);

// Nonlinear PDE Solver Functions
int nonlinear_pde_init(NonlinearPDESolver* solver, size_t spatial_dim,
                       const size_t* grid_size, NLPSolverType solver_type,
                       PDEFunction pde_func, void* params);
void nonlinear_pde_free(NonlinearPDESolver* solver);
int nonlinear_pde_solve(NonlinearPDESolver* solver, double t0, double t_end,
                       const double* u0, double* u_out);

// Distributed Nonlinear Solver Functions
int distributed_nonlinear_init(DistributedNonlinearSolver* solver,
                               NonlinearODESolver* base, size_t num_workers);
void distributed_nonlinear_free(DistributedNonlinearSolver* solver);
int distributed_nonlinear_solve(DistributedNonlinearSolver* solver,
                                ODEFunction f, double t0, double t_end,
                                const double* y0, double* y_out);

// Data-Driven Nonlinear Solver Functions
int datadriven_nonlinear_init(DataDrivenNonlinearSolver* solver,
                              NonlinearODESolver* base, size_t num_layers);
void datadriven_nonlinear_free(DataDrivenNonlinearSolver* solver);
int datadriven_nonlinear_solve(DataDrivenNonlinearSolver* solver,
                               ODEFunction f, double t0, double t_end,
                               const double* y0, double* y_out);

// Online Nonlinear Solver Functions
int online_nonlinear_init(OnlineNonlinearSolver* solver,
                          NonlinearODESolver* base, double learning_rate);
void online_nonlinear_free(OnlineNonlinearSolver* solver);
int online_nonlinear_solve(OnlineNonlinearSolver* solver, ODEFunction f,
                          double t0, double t_end, const double* y0,
                          double* y_out);

// Real-Time Nonlinear Solver Functions
int realtime_nonlinear_init(RealtimeNonlinearSolver* solver,
                            NonlinearODESolver* base,
                            void (*callback)(double t, const double* y, size_t n, void* data),
                            void* callback_data);
void realtime_nonlinear_free(RealtimeNonlinearSolver* solver);
int realtime_nonlinear_solve(RealtimeNonlinearSolver* solver, ODEFunction f,
                            double t0, double t_end, const double* y0,
                            double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* NONLINEAR_SOLVER_H */
