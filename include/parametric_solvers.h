/*
 * Parametric Solvers for Differential Equations
 * Methods with configurable and learnable parameters
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef PARAMETRIC_SOLVERS_H
#define PARAMETRIC_SOLVERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Parameter Configuration Structure
 */
typedef struct {
    double step_size;            // Fixed or initial step size
    double learning_rate;         // For adaptive methods
    double tolerance;             // Error tolerance
    size_t num_layers;            // For hierarchical methods
    size_t hidden_dim;            // For hierarchical methods
    size_t num_workers;           // For parallel methods
    double* custom_weights;       // Custom method weights
    size_t num_weights;           // Number of custom weights
    double quantum_fidelity;       // For quantum methods
    double entanglement;          // For quantum methods
    int use_learned_params;       // Whether to use learned parameters
} SolverParameters;

/**
 * Parametric Euler Solver
 * Uses configurable parameters
 */
typedef struct {
    size_t state_dim;
    SolverParameters* params;
    double* learned_step_size;   // Learned from data
    double* learned_correction;   // Learned correction term
    size_t learning_iterations;
} ParametricEulerSolver;

/**
 * Parametric RK3 Solver
 * Uses configurable and learnable parameters
 */
typedef struct {
    size_t state_dim;
    SolverParameters* params;
    double* learned_weights;     // Learned stage weights [3]
    double* learned_biases;       // Learned biases
    double* learned_step_size;
    size_t learning_iterations;
} ParametricRK3Solver;

/**
 * Parametric Adams Solver
 * Uses configurable order and weights
 */
typedef struct {
    size_t state_dim;
    SolverParameters* params;
    size_t order;                 // Configurable order
    double* learned_weights;      // Learned predictor/corrector weights
    double* learned_step_size;
    double** history;
    size_t history_size;
} ParametricAdamsSolver;

/**
 * Parametric Hierarchical RK Solver
 * Uses configurable layers and learnable weights
 */
typedef struct {
    size_t state_dim;
    SolverParameters* params;
    double** learned_layer_weights;  // Learned layer weights
    double* learned_attention_weights; // Learned attention weights
    double learned_learning_rate;
    double* learned_step_size;
    size_t learning_iterations;
} ParametricHierarchicalRKSolver;

/**
 * Parametric Parallel RK Solver
 * Uses configurable parallelization and learnable load balancing
 */
typedef struct {
    size_t state_dim;
    SolverParameters* params;
    size_t* learned_work_ranges; // Learned work distribution
    double* learned_step_size;
    double learned_sync_threshold;
} ParametricParallelRKSolver;

/**
 * Parametric Quantum SLAM Solver
 * Uses configurable quantum parameters and learnable fidelity
 */
typedef struct {
    size_t state_dim;
    SolverParameters* params;
    double learned_fidelity;      // Learned quantum fidelity
    double learned_entanglement;  // Learned entanglement strength
    double* learned_quantum_weights; // Learned quantum gate weights
    double* quantum_state;
    size_t learning_iterations;
} ParametricQuantumSLAMSolver;

// Parametric Euler Functions
int parametric_euler_init(ParametricEulerSolver* solver, size_t state_dim, 
                          const SolverParameters* params);
void parametric_euler_free(ParametricEulerSolver* solver);
int parametric_euler_learn(ParametricEulerSolver* solver, ODEFunction f,
                           double t0, double t_end, const double* y0,
                           const double* y_target, void* params);
double parametric_euler_step(ParametricEulerSolver* solver, ODEFunction f,
                              double t, double* y, void* params);
size_t parametric_euler_solve(ParametricEulerSolver* solver, ODEFunction f,
                              double t0, double t_end, const double* y0,
                              void* params, double* t_out, double* y_out);

// Parametric RK3 Functions
int parametric_rk3_init(ParametricRK3Solver* solver, size_t state_dim,
                         const SolverParameters* params);
void parametric_rk3_free(ParametricRK3Solver* solver);
int parametric_rk3_learn(ParametricRK3Solver* solver, ODEFunction f,
                          double t0, double t_end, const double* y0,
                          const double* y_target, void* params);
double parametric_rk3_step(ParametricRK3Solver* solver, ODEFunction f,
                             double t, double* y, void* params);
size_t parametric_rk3_solve(ParametricRK3Solver* solver, ODEFunction f,
                             double t0, double t_end, const double* y0,
                             void* params, double* t_out, double* y_out);

// Parametric Adams Functions
int parametric_adams_init(ParametricAdamsSolver* solver, size_t state_dim,
                          const SolverParameters* params);
void parametric_adams_free(ParametricAdamsSolver* solver);
int parametric_adams_learn(ParametricAdamsSolver* solver, ODEFunction f,
                            double t0, double t_end, const double* y0,
                            const double* y_target, void* params);
double parametric_adams_step(ParametricAdamsSolver* solver, ODEFunction f,
                              double t, double* y, void* params);
size_t parametric_adams_solve(ParametricAdamsSolver* solver, ODEFunction f,
                               double t0, double t_end, const double* y0,
                               void* params, double* t_out, double* y_out);

// Parametric Hierarchical RK Functions
int parametric_hierarchical_rk_init(ParametricHierarchicalRKSolver* solver,
                                     size_t state_dim, const SolverParameters* params);
void parametric_hierarchical_rk_free(ParametricHierarchicalRKSolver* solver);
int parametric_hierarchical_rk_learn(ParametricHierarchicalRKSolver* solver,
                                      ODEFunction f, double t0, double t_end,
                                      const double* y0, const double* y_target, void* params);
double parametric_hierarchical_rk_step(ParametricHierarchicalRKSolver* solver,
                                       ODEFunction f, double t, double* y, void* params);
size_t parametric_hierarchical_rk_solve(ParametricHierarchicalRKSolver* solver,
                                         ODEFunction f, double t0, double t_end,
                                         const double* y0, void* params,
                                         double* t_out, double* y_out);

// Parametric Parallel RK Functions
int parametric_parallel_rk_init(ParametricParallelRKSolver* solver,
                                 size_t state_dim, const SolverParameters* params);
void parametric_parallel_rk_free(ParametricParallelRKSolver* solver);
int parametric_parallel_rk_learn(ParametricParallelRKSolver* solver,
                                  ODEFunction f, double t0, double t_end,
                                  const double* y0, const double* y_target, void* params);
double parametric_parallel_rk_step(ParametricParallelRKSolver* solver,
                                    ODEFunction f, double t, double* y, void* params);
size_t parametric_parallel_rk_solve(ParametricParallelRKSolver* solver,
                                    ODEFunction f, double t0, double t_end,
                                    const double* y0, void* params,
                                    double* t_out, double* y_out);

// Parametric Quantum SLAM Functions
int parametric_quantum_slam_init(ParametricQuantumSLAMSolver* solver,
                                  size_t state_dim, const SolverParameters* params);
void parametric_quantum_slam_free(ParametricQuantumSLAMSolver* solver);
int parametric_quantum_slam_learn(ParametricQuantumSLAMSolver* solver,
                                    ODEFunction f, double t0, double t_end,
                                    const double* y0, const double* y_target, void* params);
double parametric_quantum_slam_step(ParametricQuantumSLAMSolver* solver,
                                     ODEFunction f, double t, double* y, void* params);
size_t parametric_quantum_slam_solve(ParametricQuantumSLAMSolver* solver,
                                      ODEFunction f, double t0, double t_end,
                                      const double* y0, void* params,
                                      double* t_out, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* PARAMETRIC_SOLVERS_H */
