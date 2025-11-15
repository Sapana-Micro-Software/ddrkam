/*
 * Additional Distributed, Data-Driven, Online, Real-Time Solvers
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef DISTRIBUTED_SOLVERS_H
#define DISTRIBUTED_SOLVERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "realtime_online.h"
#include "nonlinear_solver.h"

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Distributed Data-Driven Solver
 * Combines distributed computing with data-driven methods
 */
typedef struct {
    size_t state_dim;
    size_t num_workers;
    size_t num_layers;
    
    // Distributed state
    int rank;
    int size;
    void* mpi_comm;
    
    // Data-driven layers
    double** layer_weights;
    double* attention_weights;
    double learning_rate;
    
    // Work distribution
    size_t* work_ranges;
    double* local_state;
    double* global_state;
} DistributedDataDrivenSolver;

/**
 * Online Data-Driven Solver
 * Combines online learning with data-driven methods
 */
typedef struct {
    size_t state_dim;
    size_t num_layers;
    double* adaptive_step_size;
    double learning_rate;
    
    // Data-driven layers
    double** layer_weights;
    double* attention_weights;
    
    // Online adaptation
    double error_threshold;
    size_t adaptation_count;
} OnlineDataDrivenSolver;

/**
 * Real-Time Data-Driven Solver
 * Combines real-time processing with data-driven methods
 */
typedef struct {
    size_t state_dim;
    size_t num_layers;
    double step_size;
    
    // Data-driven layers
    double** layer_weights;
    double* attention_weights;
    
    // Real-time buffers
    double* buffer;
    size_t buffer_size;
    void (*callback)(double t, const double* y, size_t n, void* data);
    void* callback_data;
} RealtimeDataDrivenSolver;

/**
 * Distributed Online Solver
 * Combines distributed computing with online learning
 */
typedef struct {
    size_t state_dim;
    size_t num_workers;
    double* adaptive_step_size;
    double learning_rate;
    
    // Distributed state
    int rank;
    int size;
    size_t* work_ranges;
} DistributedOnlineSolver;

/**
 * Distributed Real-Time Solver
 * Combines distributed computing with real-time processing
 */
typedef struct {
    size_t state_dim;
    size_t num_workers;
    double step_size;
    
    // Distributed state
    int rank;
    int size;
    size_t* work_ranges;
    
    // Real-time buffers
    double* buffer;
    size_t buffer_size;
    void (*callback)(double t, const double* y, size_t n, void* data);
    void* callback_data;
} DistributedRealtimeSolver;

// Distributed Data-Driven Functions
int distributed_datadriven_init(DistributedDataDrivenSolver* solver,
                                size_t state_dim, size_t num_workers, size_t num_layers);
void distributed_datadriven_free(DistributedDataDrivenSolver* solver);
int distributed_datadriven_solve(DistributedDataDrivenSolver* solver,
                                  ODEFunction f, double t0, double t_end,
                                  const double* y0, double h, void* params,
                                  double* y_out);

// Online Data-Driven Functions
int online_datadriven_init(OnlineDataDrivenSolver* solver, size_t state_dim,
                           size_t num_layers, double initial_step_size,
                           double learning_rate);
void online_datadriven_free(OnlineDataDrivenSolver* solver);
int online_datadriven_solve(OnlineDataDrivenSolver* solver, ODEFunction f,
                           double t0, double t_end, const double* y0,
                           void* params, double* y_out);

// Real-Time Data-Driven Functions
int realtime_datadriven_init(RealtimeDataDrivenSolver* solver, size_t state_dim,
                             size_t num_layers, double step_size,
                             void (*callback)(double t, const double* y, size_t n, void* data),
                             void* callback_data);
void realtime_datadriven_free(RealtimeDataDrivenSolver* solver);
int realtime_datadriven_solve(RealtimeDataDrivenSolver* solver, ODEFunction f,
                             double t0, double t_end, const double* y0,
                             double h, void* params, double* y_out);

// Distributed Online Functions
int distributed_online_init(DistributedOnlineSolver* solver, size_t state_dim,
                           size_t num_workers, double initial_step_size,
                           double learning_rate);
void distributed_online_free(DistributedOnlineSolver* solver);
int distributed_online_solve(DistributedOnlineSolver* solver, ODEFunction f,
                             double t0, double t_end, const double* y0,
                             void* params, double* y_out);

// Distributed Real-Time Functions
int distributed_realtime_init(DistributedRealtimeSolver* solver, size_t state_dim,
                              size_t num_workers, double step_size,
                              void (*callback)(double t, const double* y, size_t n, void* data),
                              void* callback_data);
void distributed_realtime_free(DistributedRealtimeSolver* solver);
int distributed_realtime_solve(DistributedRealtimeSolver* solver, ODEFunction f,
                              double t0, double t_end, const double* y0,
                              double h, void* params, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* DISTRIBUTED_SOLVERS_H */
