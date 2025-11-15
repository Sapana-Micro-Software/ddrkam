/*
 * Extended Distributed, Data-Driven, Online, Real-Time Solvers
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef EXTENDED_DISTRIBUTED_H
#define EXTENDED_DISTRIBUTED_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "parallel_rk.h"
#include "realtime_online.h"

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Distributed Data-Driven Solver
 * Combines distributed computing with data-driven methods
 */
typedef struct {
    ParallelRKSolver* parallel_solver;
    HierarchicalRKSolver* hierarchical_solver;
    int rank;
    int size;
    void* mpi_comm;
    size_t num_layers;
    double learning_rate;
} DistributedDataDrivenSolver;

/**
 * Distributed Online Solver
 * Combines distributed computing with online learning
 */
typedef struct {
    ParallelRKSolver* parallel_solver;
    OnlineRKSolver* online_solver;
    int rank;
    int size;
    void* mpi_comm;
    double* adaptive_parameters;
} DistributedOnlineSolver;

/**
 * Distributed Real-Time Solver
 * Combines distributed computing with real-time processing
 */
typedef struct {
    ParallelRKSolver* parallel_solver;
    RealtimeRKSolver* realtime_solver;
    int rank;
    int size;
    void* mpi_comm;
    double* buffer;
    size_t buffer_size;
} DistributedRealtimeSolver;

/**
 * Data-Driven Online Solver
 * Combines data-driven methods with online learning
 */
typedef struct {
    HierarchicalRKSolver* hierarchical_solver;
    OnlineRKSolver* online_solver;
    double* combined_weights;
    size_t num_layers;
} DataDrivenOnlineSolver;

/**
 * Data-Driven Real-Time Solver
 * Combines data-driven methods with real-time processing
 */
typedef struct {
    HierarchicalRKSolver* hierarchical_solver;
    RealtimeRKSolver* realtime_solver;
    double* streaming_buffer;
    size_t buffer_size;
} DataDrivenRealtimeSolver;

/**
 * Online Real-Time Solver
 * Combines online learning with real-time processing
 */
typedef struct {
    OnlineRKSolver* online_solver;
    RealtimeRKSolver* realtime_solver;
    double* adaptive_buffer;
    size_t buffer_size;
} OnlineRealtimeSolver;

/**
 * Fully Integrated Solver
 * Combines all features: distributed, data-driven, online, real-time
 */
typedef struct {
    ParallelRKSolver* parallel_solver;
    HierarchicalRKSolver* hierarchical_solver;
    OnlineRKSolver* online_solver;
    RealtimeRKSolver* realtime_solver;
    int rank;
    int size;
    void* mpi_comm;
    double* combined_state;
    size_t num_layers;
} FullyIntegratedSolver;

// Distributed Data-Driven Functions
int distributed_datadriven_init(DistributedDataDrivenSolver* solver,
                               size_t state_dim, size_t num_workers,
                               size_t num_layers, int rank, int size);
void distributed_datadriven_free(DistributedDataDrivenSolver* solver);
int distributed_datadriven_solve(DistributedDataDrivenSolver* solver, ODEFunction f,
                                 double t0, double t_end, const double* y0,
                                 double h, void* params, double* solution);

// Distributed Online Functions
int distributed_online_init(DistributedOnlineSolver* solver,
                           size_t state_dim, size_t num_workers,
                           double learning_rate, int rank, int size);
void distributed_online_free(DistributedOnlineSolver* solver);
int distributed_online_solve(DistributedOnlineSolver* solver, ODEFunction f,
                            double t0, double t_end, const double* y0,
                            void* params, double* solution);

// Distributed Real-Time Functions
int distributed_realtime_init(DistributedRealtimeSolver* solver,
                              size_t state_dim, size_t num_workers,
                              double step_size, int rank, int size,
                              void (*callback)(double, const double*, size_t, void*),
                              void* callback_data);
void distributed_realtime_free(DistributedRealtimeSolver* solver);
int distributed_realtime_solve(DistributedRealtimeSolver* solver, ODEFunction f,
                              double t0, double t_end, const double* y0,
                              void* params, double* solution);

// Data-Driven Online Functions
int datadriven_online_init(DataDrivenOnlineSolver* solver,
                          size_t state_dim, size_t num_layers,
                          double learning_rate);
void datadriven_online_free(DataDrivenOnlineSolver* solver);
int datadriven_online_solve(DataDrivenOnlineSolver* solver, ODEFunction f,
                           double t0, double t_end, const double* y0,
                           void* params, double* solution);

// Data-Driven Real-Time Functions
int datadriven_realtime_init(DataDrivenRealtimeSolver* solver,
                            size_t state_dim, size_t num_layers,
                            double step_size,
                            void (*callback)(double, const double*, size_t, void*),
                            void* callback_data);
void datadriven_realtime_free(DataDrivenRealtimeSolver* solver);
int datadriven_realtime_solve(DataDrivenRealtimeSolver* solver, ODEFunction f,
                             double t0, double t_end, const double* y0,
                             void* params, double* solution);

// Online Real-Time Functions
int online_realtime_init(OnlineRealtimeSolver* solver,
                        size_t state_dim, double initial_step_size,
                        double learning_rate,
                        void (*callback)(double, const double*, size_t, void*),
                        void* callback_data);
void online_realtime_free(OnlineRealtimeSolver* solver);
int online_realtime_solve(OnlineRealtimeSolver* solver, ODEFunction f,
                         double t0, double t_end, const double* y0,
                         void* params, double* solution);

// Fully Integrated Solver Functions
int fully_integrated_init(FullyIntegratedSolver* solver,
                          size_t state_dim, size_t num_workers,
                          size_t num_layers, double learning_rate,
                          int rank, int size,
                          void (*callback)(double, const double*, size_t, void*),
                          void* callback_data);
void fully_integrated_free(FullyIntegratedSolver* solver);
int fully_integrated_solve(FullyIntegratedSolver* solver, ODEFunction f,
                          double t0, double t_end, const double* y0,
                          void* params, double* solution);

#ifdef __cplusplus
}
#endif

#endif /* EXTENDED_DISTRIBUTED_H */
