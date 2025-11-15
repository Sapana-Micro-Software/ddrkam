/*
 * Parallel and Distributed Adams Methods Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef PARALLEL_ADAMS_H
#define PARALLEL_ADAMS_H

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
 * Parallel Adams Solver
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
    
    // Adams history
    double** history_t;      // Time history
    double** history_y;       // State history
    size_t history_size;
} ParallelAdamsSolver;

/**
 * Initialize parallel Adams solver
 */
int parallel_adams_init(ParallelAdamsSolver* solver, size_t state_dim,
                        size_t num_workers, ParallelMode mode,
                        StackedConfig* stacked);

/**
 * Free parallel Adams solver
 */
void parallel_adams_free(ParallelAdamsSolver* solver);

/**
 * Parallel Adams step with distributed execution
 */
double parallel_adams_step(ParallelAdamsSolver* solver, ODEFunction f,
                          double t, double* y, double h, void* params);

/**
 * Stacked/Hierarchical Adams step
 */
double stacked_adams_step(ParallelAdamsSolver* solver, ODEFunction f,
                          double t, double* y, double h, void* params);

/**
 * Concurrent Adams execution
 */
int concurrent_adams_execute(ParallelAdamsSolver* solvers[], size_t num_solvers,
                             ODEFunction f, double t, const double* y, double h,
                             void* params, double** results);

/**
 * Solve ODE using parallel Adams over time interval
 */
size_t parallel_adams_solve(ParallelAdamsSolver* solver, ODEFunction f,
                            double t0, double t_end, const double* y0,
                            double h, void* params, double* t_out, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* PARALLEL_ADAMS_H */
