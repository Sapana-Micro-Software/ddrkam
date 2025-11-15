/*
 * Parallel and Distributed Euler Methods Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef PARALLEL_EULER_H
#define PARALLEL_EULER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "parallel_rk.h"  // For ParallelMode and StackedConfig

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Parallel Euler Solver
 */
typedef struct {
    size_t state_dim;
    size_t num_workers;
    ParallelMode mode;
    StackedConfig* stacked;
    
    // Distributed state
    int rank;
    int size;
    void* mpi_comm;
    
    // Threading state
    void** threads;
    int num_threads;
    
    // Work distribution
    size_t* work_ranges;
    double* local_state;
    double* global_state;
} ParallelEulerSolver;

/**
 * Initialize parallel Euler solver
 */
int parallel_euler_init(ParallelEulerSolver* solver, size_t state_dim,
                        size_t num_workers, ParallelMode mode,
                        StackedConfig* stacked);

/**
 * Free parallel Euler solver
 */
void parallel_euler_free(ParallelEulerSolver* solver);

/**
 * Parallel Euler step with distributed execution
 */
double parallel_euler_step(ParallelEulerSolver* solver, ODEFunction f,
                          double t, double* y, double h, void* params);

/**
 * Stacked/Hierarchical Euler step
 */
double stacked_euler_step(ParallelEulerSolver* solver, ODEFunction f,
                          double t, double* y, double h, void* params);

/**
 * Concurrent Euler execution
 */
int concurrent_euler_execute(ParallelEulerSolver* solvers[], size_t num_solvers,
                             ODEFunction f, double t, const double* y, double h,
                             void* params, double** results);

/**
 * Solve ODE using parallel Euler over time interval
 */
size_t parallel_euler_solve(ParallelEulerSolver* solver, ODEFunction f,
                            double t0, double t_end, const double* y0,
                            double h, void* params, double* t_out, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* PARALLEL_EULER_H */
