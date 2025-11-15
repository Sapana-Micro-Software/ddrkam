/*
 * Optimization-Based Solvers: ADAM, AdaGrad, and Karmarkar's Algorithm
 * For nonlinear, nonconvex, and online differential equations (ODEs & PDEs)
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef OPTIMIZATION_SOLVERS_H
#define OPTIMIZATION_SOLVERS_H

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
 * Objective function for optimization
 */
typedef double (*ObjectiveFunction)(const double* x, size_t n, void* params);

/**
 * Gradient function
 */
typedef void (*GradientFunction)(const double* x, size_t n, double* grad, void* params);

/**
 * ADAM Solver Type
 */
typedef enum {
    ADAM_ODE,      // ADAM for ODEs
    ADAM_PDE,      // ADAM for PDEs
    ADAM_ONLINE    // ADAM for online learning
} ADAMSolverType;

/**
 * ADAM Solver for ODEs/PDEs
 * Adaptive Moment Estimation for nonlinear nonconvex optimization
 */
typedef struct {
    size_t state_dim;
    ADAMSolverType solver_type;
    
    // ADAM parameters
    double alpha;           // Learning rate (step size)
    double beta1;           // Exponential decay rate for first moment (typically 0.9)
    double beta2;           // Exponential decay rate for second moment (typically 0.999)
    double epsilon;         // Small constant for numerical stability (typically 1e-8)
    
    // Moment estimates
    double* m;              // First moment estimate (mean)
    double* v;              // Second moment estimate (variance)
    
    // State and gradients
    double* state;
    double* gradient;
    double* previous_gradient;
    
    // Time step tracking
    size_t t;               // Time step counter
    
    // Convergence
    double tolerance;
    size_t max_iterations;
    size_t current_iteration;
    
    // For PDEs
    size_t spatial_dim;
    size_t* grid_size;
    double* spatial_state;
    
    // Online learning
    double* buffer;
    size_t buffer_size;
    size_t buffer_index;
} ADAMSolver;

/**
 * AdaGrad Solver for ODEs/PDEs
 * Adaptive Gradient Algorithm for sparse gradients
 */
typedef struct {
    size_t state_dim;
    ADAMSolverType solver_type;
    
    // AdaGrad parameters
    double learning_rate;   // Initial learning rate
    double epsilon;         // Small constant (typically 1e-8)
    
    // Accumulated squared gradients
    double* G;              // Sum of squared gradients
    
    // State and gradients
    double* state;
    double* gradient;
    double* previous_gradient;
    
    // Convergence
    double tolerance;
    size_t max_iterations;
    size_t current_iteration;
    
    // For PDEs
    size_t spatial_dim;
    size_t* grid_size;
    double* spatial_state;
    
    // Online learning
    double* buffer;
    size_t buffer_size;
    size_t buffer_index;
} AdaGradSolver;

/**
 * Karmarkar's Algorithm Solver
 * Interior point method for linear/nonlinear programming formulation of ODEs/PDEs
 */
typedef struct {
    size_t state_dim;
    ADAMSolverType solver_type;
    
    // Karmarkar's algorithm parameters
    double alpha;           // Step size parameter (typically 0.25)
    double beta;            // Barrier parameter reduction factor (typically 0.5)
    double mu;              // Barrier parameter
    double epsilon;         // Convergence tolerance
    
    // Problem formulation
    double* c;              // Objective function coefficients
    double** A;             // Constraint matrix
    double* b;              // Constraint bounds
    size_t num_constraints;
    
    // Interior point state
    double* x;              // Current point (must be in interior)
    double* y;              // Dual variables
    double* s;              // Slack variables
    
    // Search direction
    double* dx;
    double* dy;
    double* ds;
    
    // Convergence
    double tolerance;
    size_t max_iterations;
    size_t current_iteration;
    
    // For PDEs
    size_t spatial_dim;
    size_t* grid_size;
    double* spatial_state;
    
    // Online/adaptive
    double* buffer;
    size_t buffer_size;
} KarmarkarSolver;

// ADAM Solver Functions
int adam_solver_init(ADAMSolver* solver, size_t state_dim, ADAMSolverType type,
                     double alpha, double beta1, double beta2, double epsilon);
void adam_solver_free(ADAMSolver* solver);
int adam_ode_solve(ADAMSolver* solver, ODEFunction f, double t0, double t_end,
                   const double* y0, void* params, double* y_out);
int adam_pde_solve(ADAMSolver* solver, PDEFunction f, double t0, double t_end,
                   const double* u0, size_t spatial_dim, const size_t* grid_size,
                   void* params, double* u_out);
int adam_online_step(ADAMSolver* solver, ODEFunction f, double t, double* y,
                     void* params);

// AdaGrad Solver Functions
int adagrad_solver_init(AdaGradSolver* solver, size_t state_dim, ADAMSolverType type,
                        double learning_rate, double epsilon);
void adagrad_solver_free(AdaGradSolver* solver);
int adagrad_ode_solve(AdaGradSolver* solver, ODEFunction f, double t0, double t_end,
                      const double* y0, void* params, double* y_out);
int adagrad_pde_solve(AdaGradSolver* solver, PDEFunction f, double t0, double t_end,
                      const double* u0, size_t spatial_dim, const size_t* grid_size,
                      void* params, double* u_out);
int adagrad_online_step(AdaGradSolver* solver, ODEFunction f, double t, double* y,
                        void* params);

// Karmarkar's Algorithm Functions
int karmarkar_solver_init(KarmarkarSolver* solver, size_t state_dim, ADAMSolverType type,
                           double alpha, double beta, double mu, double epsilon,
                           const double* c, const double** A, const double* b,
                           size_t num_constraints);
void karmarkar_solver_free(KarmarkarSolver* solver);
int karmarkar_ode_solve(KarmarkarSolver* solver, ODEFunction f, double t0, double t_end,
                        const double* y0, void* params, double* y_out);
int karmarkar_pde_solve(KarmarkarSolver* solver, PDEFunction f, double t0, double t_end,
                        const double* u0, size_t spatial_dim, const size_t* grid_size,
                        void* params, double* u_out);
int karmarkar_online_step(KarmarkarSolver* solver, ODEFunction f, double t, double* y,
                          void* params);

// Helper functions for gradient computation
void compute_ode_gradient(ODEFunction f, double t, const double* y, size_t n,
                          void* params, double* gradient);
void compute_pde_gradient(PDEFunction f, double t, double x, double y,
                          const double* u, size_t n, void* params, double* gradient);

#ifdef __cplusplus
}
#endif

#endif /* OPTIMIZATION_SOLVERS_H */
