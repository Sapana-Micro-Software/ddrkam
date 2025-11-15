/*
 * Hybrid Distributed, Data-Driven, Online, Real-Time Solvers
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef HYBRID_SOLVERS_H
#define HYBRID_SOLVERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "parallel_rk.h"
#include "realtime_online.h"

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Data callback for streaming/online updates
 */
typedef void (*DataCallback)(double t, const double* y, size_t n, void* user_data);

/**
 * Execution mode flags
 */
typedef enum {
    MODE_STANDARD = 0,
    MODE_DATA_DRIVEN = 1 << 0,
    MODE_DISTRIBUTED = 1 << 1,
    MODE_ONLINE = 1 << 2,
    MODE_REALTIME = 1 << 3,
    MODE_PARALLEL = 1 << 4,
    MODE_STACKED = 1 << 5,
    MODE_DYNAMIC = 1 << 6
} SolverMode;

/**
 * Distributed Data-Driven RK3 Solver
 * Combines distributed computing with hierarchical data-driven architecture
 */
typedef struct {
    size_t state_dim;
    size_t num_workers;
    ParallelMode parallel_mode;
    StackedConfig* stacked_config;
    
    // Distributed state
    int rank;
    int size;
    void* mpi_comm;
    
    // Data-driven components
    double* attention_weights;
    double* layer_weights;
    size_t num_layers;
    size_t hidden_dim;
    double learning_rate;
    
    // Performance metrics
    double avg_step_time;
    uint64_t total_steps;
} DistributedDataDrivenRKSolver;

/**
 * Online Data-Driven RK3 Solver
 * Combines online adaptive learning with data-driven architecture
 */
typedef struct {
    size_t state_dim;
    double* adaptive_step_size;
    double learning_rate;
    
    // Data-driven components
    double* attention_weights;
    double** layer_weights;
    double** layer_biases;
    size_t num_layers;
    size_t hidden_dim;
    
    // Online learning
    double* weight_history;
    size_t history_size;
    size_t history_idx;
    double error_threshold;
    double min_step_size;
    double max_step_size;
    
    // Performance tracking
    double cumulative_error;
    size_t adaptation_count;
} OnlineDataDrivenRKSolver;

/**
 * Real-Time Data-Driven RK3 Solver
 * Combines real-time streaming with data-driven architecture
 */
typedef struct {
    size_t state_dim;
    double step_size;
    double current_time;
    double* current_state;
    
    // Streaming buffer
    double* buffer;
    size_t buffer_size;
    size_t buffer_idx;
    
    // Data-driven components
    double* attention_weights;
    double** layer_weights;
    size_t num_layers;
    size_t hidden_dim;
    
    // Real-time callback
    DataCallback callback;
    void* callback_data;
    
    // Performance metrics
    double avg_step_time;
    uint64_t total_steps;
} RealtimeDataDrivenRKSolver;

/**
 * Distributed Online RK3 Solver
 * Combines distributed computing with online adaptive learning
 */
typedef struct {
    size_t state_dim;
    size_t num_workers;
    ParallelMode parallel_mode;
    double* adaptive_step_size;
    double learning_rate;
    
    // Distributed state
    int rank;
    int size;
    void* mpi_comm;
    
    // Online learning
    double error_threshold;
    double min_step_size;
    double max_step_size;
    double cumulative_error;
} DistributedOnlineRKSolver;

/**
 * Distributed Real-Time RK3 Solver
 * Combines distributed computing with real-time streaming
 */
typedef struct {
    size_t state_dim;
    size_t num_workers;
    ParallelMode parallel_mode;
    double step_size;
    
    // Distributed state
    int rank;
    int size;
    void* mpi_comm;
    
    // Real-time components
    double* buffer;
    size_t buffer_size;
    DataCallback callback;
    void* callback_data;
} DistributedRealtimeRKSolver;

/**
 * Hybrid Multi-Mode RK3 Solver
 * Combines multiple execution modes simultaneously
 */
typedef struct {
    size_t state_dim;
    SolverMode modes;
    
    // Configuration
    size_t num_workers;
    ParallelMode parallel_mode;
    StackedConfig* stacked_config;
    double learning_rate;
    double adaptation_rate;
    
    // State
    double current_time;
    double* current_state;
    double* adaptive_step_size;
    
    // Components (conditionally allocated based on modes)
    void* parallel_solver;
    void* online_solver;
    void* realtime_solver;
    void* data_driven_solver;
    
    // Performance
    double avg_step_time;
    uint64_t total_steps;
} HybridRKSolver;

// Distributed Data-Driven Functions
int distributed_dd_rk_init(DistributedDataDrivenRKSolver* solver, size_t state_dim,
                           size_t num_workers, ParallelMode mode, size_t num_layers,
                           size_t hidden_dim);
void distributed_dd_rk_free(DistributedDataDrivenRKSolver* solver);
double distributed_dd_rk_step(DistributedDataDrivenRKSolver* solver, ODEFunction f,
                              double t, double* y, double h, void* params);

// Online Data-Driven Functions
int online_dd_rk_init(OnlineDataDrivenRKSolver* solver, size_t state_dim,
                     double initial_step_size, double learning_rate,
                     size_t num_layers, size_t hidden_dim);
void online_dd_rk_free(OnlineDataDrivenRKSolver* solver);
double online_dd_rk_step(OnlineDataDrivenRKSolver* solver, ODEFunction f,
                         double t, double* y, void* params);

// Real-Time Data-Driven Functions
int realtime_dd_rk_init(RealtimeDataDrivenRKSolver* solver, size_t state_dim,
                       double step_size, DataCallback callback, void* callback_data,
                       size_t num_layers, size_t hidden_dim);
void realtime_dd_rk_free(RealtimeDataDrivenRKSolver* solver);
double realtime_dd_rk_step(RealtimeDataDrivenRKSolver* solver, ODEFunction f,
                           double t, double* y, double h, void* params);

// Distributed Online Functions
int distributed_online_rk_init(DistributedOnlineRKSolver* solver, size_t state_dim,
                               size_t num_workers, ParallelMode mode,
                               double initial_step_size, double learning_rate);
void distributed_online_rk_free(DistributedOnlineRKSolver* solver);
double distributed_online_rk_step(DistributedOnlineRKSolver* solver, ODEFunction f,
                                   double t, double* y, void* params);

// Distributed Real-Time Functions
int distributed_realtime_rk_init(DistributedRealtimeRKSolver* solver, size_t state_dim,
                                size_t num_workers, ParallelMode mode, double step_size,
                                DataCallback callback, void* callback_data);
void distributed_realtime_rk_free(DistributedRealtimeRKSolver* solver);
double distributed_realtime_rk_step(DistributedRealtimeRKSolver* solver, ODEFunction f,
                                    double t, double* y, double h, void* params);

// Hybrid Multi-Mode Functions
int hybrid_rk_init(HybridRKSolver* solver, size_t state_dim, SolverMode modes,
                   size_t num_workers, ParallelMode parallel_mode,
                   StackedConfig* stacked_config, double learning_rate);
void hybrid_rk_free(HybridRKSolver* solver);
double hybrid_rk_step(HybridRKSolver* solver, ODEFunction f, double t,
                     double* y, double h, void* params);

// Similar structures and functions for Adams Methods
typedef struct {
    size_t state_dim;
    size_t num_workers;
    ParallelMode parallel_mode;
    StackedConfig* stacked_config;
    double* attention_weights;
    size_t num_layers;
    size_t hidden_dim;
    double learning_rate;
    int rank;
    int size;
    void* mpi_comm;
    double** history_t;
    double** history_y;
    size_t history_size;
} DistributedDataDrivenAdamsSolver;

typedef struct {
    size_t state_dim;
    double* adaptive_step_size;
    double learning_rate;
    double* attention_weights;
    double** layer_weights;
    size_t num_layers;
    size_t hidden_dim;
    double** history_t;
    double** history_y;
    size_t history_size;
    double error_threshold;
} OnlineDataDrivenAdamsSolver;

// Adams Methods Functions
int distributed_dd_adams_init(DistributedDataDrivenAdamsSolver* solver, size_t state_dim,
                              size_t num_workers, ParallelMode mode, size_t num_layers,
                              size_t hidden_dim);
void distributed_dd_adams_free(DistributedDataDrivenAdamsSolver* solver);
double distributed_dd_adams_step(DistributedDataDrivenAdamsSolver* solver, ODEFunction f,
                                double t, double* y, double h, void* params);

int online_dd_adams_init(OnlineDataDrivenAdamsSolver* solver, size_t state_dim,
                        double initial_step_size, double learning_rate,
                        size_t num_layers, size_t hidden_dim);
void online_dd_adams_free(OnlineDataDrivenAdamsSolver* solver);
double online_dd_adams_step(OnlineDataDrivenAdamsSolver* solver, ODEFunction f,
                           double t, double* y, void* params);

// Similar structures for Euler Methods
typedef struct {
    size_t state_dim;
    size_t num_workers;
    ParallelMode parallel_mode;
    StackedConfig* stacked_config;
    double* attention_weights;
    size_t num_layers;
    size_t hidden_dim;
    double learning_rate;
    int rank;
    int size;
} DistributedDataDrivenEulerSolver;

typedef struct {
    size_t state_dim;
    double* adaptive_step_size;
    double learning_rate;
    double* attention_weights;
    double** layer_weights;
    size_t num_layers;
    size_t hidden_dim;
    double error_threshold;
} OnlineDataDrivenEulerSolver;

// Euler Methods Functions
int distributed_dd_euler_init(DistributedDataDrivenEulerSolver* solver, size_t state_dim,
                              size_t num_workers, ParallelMode mode, size_t num_layers,
                              size_t hidden_dim);
void distributed_dd_euler_free(DistributedDataDrivenEulerSolver* solver);
double distributed_dd_euler_step(DistributedDataDrivenEulerSolver* solver, ODEFunction f,
                                 double t, double* y, double h, void* params);

int online_dd_euler_init(OnlineDataDrivenEulerSolver* solver, size_t state_dim,
                        double initial_step_size, double learning_rate,
                        size_t num_layers, size_t hidden_dim);
void online_dd_euler_free(OnlineDataDrivenEulerSolver* solver);
double online_dd_euler_step(OnlineDataDrivenEulerSolver* solver, ODEFunction f,
                           double t, double* y, void* params);

#ifdef __cplusplus
}
#endif

#endif /* HYBRID_SOLVERS_H */
