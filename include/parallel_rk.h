/*
 * Parallel and Distributed Runge-Kutta Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef PARALLEL_RK_H
#define PARALLEL_RK_H

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
 * Parallel execution mode
 */
typedef enum {
    PARALLEL_OPENMP,      // OpenMP threading
    PARALLEL_PTHREAD,     // POSIX threads
    PARALLEL_MPI,         // MPI distributed
    PARALLEL_HYBRID       // Hybrid (MPI + OpenMP)
} ParallelMode;

/**
 * Hierarchical/Stacked layer configuration
 */
typedef struct {
    size_t num_layers;           // Number of stacked layers
    size_t* layer_dims;          // Dimension of each layer
    size_t hidden_dim;           // Hidden dimension for attention
    double learning_rate;        // Learning rate for adaptive refinement
    int use_attention;            // Enable attention mechanisms
    int use_residual;             // Enable residual connections
} StackedConfig;

/**
 * Parallel RK3 Solver
 */
typedef struct {
    size_t state_dim;            // State dimension
    size_t num_workers;          // Number of parallel workers
    ParallelMode mode;            // Parallel execution mode
    StackedConfig* stacked;      // Stacked/hierarchical configuration
    
    // Distributed state
    int rank;                    // MPI rank (if using MPI)
    int size;                    // MPI size (if using MPI)
    void* mpi_comm;             // MPI communicator
    
    // Threading state
    void** threads;             // Thread handles (pthread)
    int num_threads;             // Number of threads
    
    // Work distribution
    size_t* work_ranges;         // Work range per worker [start, end]
    double* local_state;         // Local state per worker
    double* global_state;        // Global aggregated state
} ParallelRKSolver;

/**
 * Initialize parallel RK3 solver
 * 
 * @param solver: Solver to initialize
 * @param state_dim: Dimension of ODE system
 * @param num_workers: Number of parallel workers
 * @param mode: Parallel execution mode
 * @param stacked: Stacked configuration (NULL for standard parallel)
 * @return: 0 on success, -1 on error
 */
int parallel_rk_init(ParallelRKSolver* solver, size_t state_dim, 
                     size_t num_workers, ParallelMode mode, 
                     StackedConfig* stacked);

/**
 * Free parallel RK3 solver
 */
void parallel_rk_free(ParallelRKSolver* solver);

/**
 * Parallel RK3 step with distributed execution
 * 
 * @param solver: Initialized solver
 * @param f: ODE function
 * @param t: Current time
 * @param y: Current state (distributed across workers)
 * @param h: Step size
 * @param params: User parameters
 * @return: New time (t + h)
 */
double parallel_rk_step(ParallelRKSolver* solver, ODEFunction f,
                        double t, double* y, double h, void* params);

/**
 * Stacked/Hierarchical RK3 step
 * Processes through multiple stacked layers with attention
 * 
 * @param solver: Initialized solver with stacked config
 * @param f: ODE function
 * @param t: Current time
 * @param y: Current state
 * @param h: Step size
 * @param params: User parameters
 * @return: New time (t + h)
 */
double stacked_rk_step(ParallelRKSolver* solver, ODEFunction f,
                      double t, double* y, double h, void* params);

/**
 * Concurrent execution: Run multiple methods simultaneously
 * 
 * @param solvers: Array of solvers (one per method)
 * @param num_solvers: Number of solvers
 * @param f: ODE function
 * @param t: Current time
 * @param y: Current state
 * @param h: Step size
 * @param params: User parameters
 * @param results: Output array for results from each solver
 * @return: 0 on success, -1 on error
 */
int concurrent_rk_execute(ParallelRKSolver* solvers[], size_t num_solvers,
                          ODEFunction f, double t, const double* y, double h,
                          void* params, double** results);

/**
 * Solve ODE using parallel RK3 over time interval
 * 
 * @param solver: Initialized solver
 * @param f: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial state
 * @param h: Step size
 * @param params: User parameters
 * @param t_out: Output time array
 * @param y_out: Output state array
 * @return: Number of steps
 */
size_t parallel_rk_solve(ParallelRKSolver* solver, ODEFunction f,
                         double t0, double t_end, const double* y0,
                         double h, void* params, double* t_out, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* PARALLEL_RK_H */
