/*
 * Non-Parametric Solvers for Differential Equations
 * Adaptive, parameter-free methods that automatically adjust to problem structure
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef NONPARAMETRIC_SOLVERS_H
#define NONPARAMETRIC_SOLVERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Non-Parametric Euler Solver
 * Automatically adapts step size and method parameters
 */
typedef struct {
    size_t state_dim;
    double* adaptive_step_size;  // Automatically adjusted
    double* error_estimate;      // Local error estimate
    double tolerance;            // Target error tolerance (auto-set if 0)
    size_t adaptation_count;
} NonParametricEulerSolver;

/**
 * Non-Parametric RK3 Solver
 * Adaptive RK3 with automatic parameter selection
 */
typedef struct {
    size_t state_dim;
    double* adaptive_step_size;
    double* error_estimate;
    double tolerance;
    double* stage_weights;       // Automatically optimized weights
    size_t adaptation_count;
} NonParametricRK3Solver;

/**
 * Non-Parametric Adams Solver
 * Adaptive Adams methods with automatic order selection
 */
typedef struct {
    size_t state_dim;
    double* adaptive_step_size;
    double* error_estimate;
    double tolerance;
    size_t adaptive_order;       // Automatically selected order (1-5)
    double** history;            // Solution history
    size_t history_size;
} NonParametricAdamsSolver;

/**
 * Non-Parametric Hierarchical RK Solver
 * Adaptive hierarchical method with automatic layer selection
 */
typedef struct {
    size_t state_dim;
    double* adaptive_step_size;
    double* error_estimate;
    double tolerance;
    size_t adaptive_layers;      // Automatically selected layers
    size_t adaptive_hidden_dim;  // Automatically selected hidden dimension
    double** layer_weights;      // Automatically initialized
    double learning_rate;        // Automatically tuned
} NonParametricHierarchicalRKSolver;

/**
 * Non-Parametric Parallel RK Solver
 * Adaptive parallel method with automatic worker allocation
 */
typedef struct {
    size_t state_dim;
    double* adaptive_step_size;
    double* error_estimate;
    double tolerance;
    size_t adaptive_workers;     // Automatically selected worker count
    size_t* work_ranges;         // Automatically balanced
} NonParametricParallelRKSolver;

/**
 * Non-Parametric Quantum SLAM Solver
 * Adaptive quantum simulation with automatic fidelity tuning
 */
typedef struct {
    size_t state_dim;
    double* adaptive_step_size;
    double* error_estimate;
    double tolerance;
    double adaptive_fidelity;    // Automatically tuned quantum fidelity
    double adaptive_entanglement; // Automatically tuned entanglement
    double* quantum_state;       // Quantum state vector
} NonParametricQuantumSLAMSolver;

// Non-Parametric Euler Functions
int nonparametric_euler_init(NonParametricEulerSolver* solver, size_t state_dim, double tolerance);
void nonparametric_euler_free(NonParametricEulerSolver* solver);
double nonparametric_euler_step(NonParametricEulerSolver* solver, ODEFunction f, 
                                double t, double* y, void* params);
size_t nonparametric_euler_solve(NonParametricEulerSolver* solver, ODEFunction f,
                                 double t0, double t_end, const double* y0,
                                 void* params, double* t_out, double* y_out);

// Non-Parametric RK3 Functions
int nonparametric_rk3_init(NonParametricRK3Solver* solver, size_t state_dim, double tolerance);
void nonparametric_rk3_free(NonParametricRK3Solver* solver);
double nonparametric_rk3_step(NonParametricRK3Solver* solver, ODEFunction f,
                               double t, double* y, void* params);
size_t nonparametric_rk3_solve(NonParametricRK3Solver* solver, ODEFunction f,
                                double t0, double t_end, const double* y0,
                                void* params, double* t_out, double* y_out);

// Non-Parametric Adams Functions
int nonparametric_adams_init(NonParametricAdamsSolver* solver, size_t state_dim, double tolerance);
void nonparametric_adams_free(NonParametricAdamsSolver* solver);
double nonparametric_adams_step(NonParametricAdamsSolver* solver, ODEFunction f,
                                 double t, double* y, void* params);
size_t nonparametric_adams_solve(NonParametricAdamsSolver* solver, ODEFunction f,
                                  double t0, double t_end, const double* y0,
                                  void* params, double* t_out, double* y_out);

// Non-Parametric Hierarchical RK Functions
int nonparametric_hierarchical_rk_init(NonParametricHierarchicalRKSolver* solver, 
                                       size_t state_dim, double tolerance);
void nonparametric_hierarchical_rk_free(NonParametricHierarchicalRKSolver* solver);
double nonparametric_hierarchical_rk_step(NonParametricHierarchicalRKSolver* solver,
                                           ODEFunction f, double t, double* y, void* params);
size_t nonparametric_hierarchical_rk_solve(NonParametricHierarchicalRKSolver* solver,
                                            ODEFunction f, double t0, double t_end,
                                            const double* y0, void* params,
                                            double* t_out, double* y_out);

// Non-Parametric Parallel RK Functions
int nonparametric_parallel_rk_init(NonParametricParallelRKSolver* solver,
                                   size_t state_dim, double tolerance);
void nonparametric_parallel_rk_free(NonParametricParallelRKSolver* solver);
double nonparametric_parallel_rk_step(NonParametricParallelRKSolver* solver,
                                       ODEFunction f, double t, double* y, void* params);
size_t nonparametric_parallel_rk_solve(NonParametricParallelRKSolver* solver,
                                        ODEFunction f, double t0, double t_end,
                                        const double* y0, void* params,
                                        double* t_out, double* y_out);

// Non-Parametric Quantum SLAM Functions
int nonparametric_quantum_slam_init(NonParametricQuantumSLAMSolver* solver,
                                     size_t state_dim, double tolerance);
void nonparametric_quantum_slam_free(NonParametricQuantumSLAMSolver* solver);
double nonparametric_quantum_slam_step(NonParametricQuantumSLAMSolver* solver,
                                         ODEFunction f, double t, double* y, void* params);
size_t nonparametric_quantum_slam_solve(NonParametricQuantumSLAMSolver* solver,
                                         ODEFunction f, double t0, double t_end,
                                         const double* y0, void* params,
                                         double* t_out, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* NONPARAMETRIC_SOLVERS_H */
