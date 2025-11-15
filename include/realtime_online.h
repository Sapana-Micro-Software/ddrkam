/*
 * Real-Time, Online, and Dynamic Numerical Methods
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef REALTIME_ONLINE_H
#define REALTIME_ONLINE_H

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
 * Data callback for streaming/online updates
 */
typedef void (*DataCallback)(double t, const double* y, size_t n, void* user_data);

/**
 * Real-Time RK3 Solver
 * Processes streaming data with minimal latency
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double step_size;
    
    // Streaming buffer
    double* buffer;
    size_t buffer_size;
    size_t buffer_idx;
    
    // Real-time callback
    DataCallback callback;
    void* callback_data;
    
    // Performance metrics
    uint64_t total_steps;
    double avg_step_time;
} RealtimeRKSolver;

/**
 * Online RK3 Solver
 * Adapts to incoming data with incremental learning
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double* adaptive_step_size;
    
    // Online learning parameters
    double learning_rate;
    double* weight_history;
    size_t history_size;
    size_t history_idx;
    
    // Adaptive parameters
    double error_threshold;
    double min_step_size;
    double max_step_size;
    
    // Performance tracking
    double cumulative_error;
    size_t adaptation_count;
} OnlineRKSolver;

/**
 * Dynamic RK3 Solver
 * Dynamic step size and parameter adaptation
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double* dynamic_step_size;
    
    // Dynamic adaptation
    double* error_estimate;
    double* stability_estimate;
    double adaptation_rate;
    
    // Parameter history
    double** parameter_history;
    size_t history_size;
    size_t history_idx;
    
    // Dynamic thresholds
    double error_tolerance;
    double stability_tolerance;
    int adaptive_mode;
} DynamicRKSolver;

/**
 * Real-Time Adams Methods Solver
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double step_size;
    
    // History for multi-step
    double** history_t;
    double** history_y;
    size_t history_size;
    size_t history_count;
    
    // Streaming buffer
    double* buffer;
    size_t buffer_size;
    
    DataCallback callback;
    void* callback_data;
} RealtimeAdamsSolver;

/**
 * Online Adams Methods Solver
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double* adaptive_step_size;
    
    // History
    double** history_t;
    double** history_y;
    size_t history_size;
    
    // Online learning
    double learning_rate;
    double error_threshold;
    double min_step_size;
    double max_step_size;
} OnlineAdamsSolver;

/**
 * Dynamic Adams Methods Solver
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double* dynamic_step_size;
    
    // History
    double** history_t;
    double** history_y;
    size_t history_size;
    
    // Dynamic adaptation
    double* error_estimate;
    double adaptation_rate;
    double error_tolerance;
} DynamicAdamsSolver;

/**
 * Real-Time Euler Solver
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double step_size;
    
    // Streaming buffer
    double* buffer;
    size_t buffer_size;
    size_t buffer_idx;
    
    DataCallback callback;
    void* callback_data;
} RealtimeEulerSolver;

/**
 * Online Euler Solver
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double* adaptive_step_size;
    
    // Online learning
    double learning_rate;
    double error_threshold;
    double min_step_size;
    double max_step_size;
} OnlineEulerSolver;

/**
 * Dynamic Euler Solver
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double* dynamic_step_size;
    
    // Dynamic adaptation
    double* error_estimate;
    double adaptation_rate;
    double error_tolerance;
} DynamicEulerSolver;

// Real-Time RK3 Functions
int realtime_rk_init(RealtimeRKSolver* solver, size_t state_dim, double step_size,
                     DataCallback callback, void* callback_data);
void realtime_rk_free(RealtimeRKSolver* solver);
double realtime_rk_step(RealtimeRKSolver* solver, ODEFunction f, double t,
                       double* y, double h, void* params);
int realtime_rk_process_stream(RealtimeRKSolver* solver, ODEFunction f,
                               const double* stream_data, size_t stream_length,
                               void* params);

// Online RK3 Functions
int online_rk_init(OnlineRKSolver* solver, size_t state_dim, double initial_step_size,
                  double learning_rate);
void online_rk_free(OnlineRKSolver* solver);
double online_rk_step(OnlineRKSolver* solver, ODEFunction f, double t,
                     double* y, void* params);
double online_rk_adapt_step_size(OnlineRKSolver* solver, double error_estimate);

// Dynamic RK3 Functions
int dynamic_rk_init(DynamicRKSolver* solver, size_t state_dim, double initial_step_size,
                   double adaptation_rate);
void dynamic_rk_free(DynamicRKSolver* solver);
double dynamic_rk_step(DynamicRKSolver* solver, ODEFunction f, double t,
                      double* y, void* params);
void dynamic_rk_adapt(DynamicRKSolver* solver, double error_estimate, double stability_estimate);

// Real-Time Adams Functions
int realtime_adams_init(RealtimeAdamsSolver* solver, size_t state_dim, double step_size,
                        DataCallback callback, void* callback_data);
void realtime_adams_free(RealtimeAdamsSolver* solver);
double realtime_adams_step(RealtimeAdamsSolver* solver, ODEFunction f, double t,
                          double* y, double h, void* params);

// Online Adams Functions
int online_adams_init(OnlineAdamsSolver* solver, size_t state_dim, double initial_step_size,
                     double learning_rate);
void online_adams_free(OnlineAdamsSolver* solver);
double online_adams_step(OnlineAdamsSolver* solver, ODEFunction f, double t,
                        double* y, void* params);

// Dynamic Adams Functions
int dynamic_adams_init(DynamicAdamsSolver* solver, size_t state_dim, double initial_step_size,
                      double adaptation_rate);
void dynamic_adams_free(DynamicAdamsSolver* solver);
double dynamic_adams_step(DynamicAdamsSolver* solver, ODEFunction f, double t,
                         double* y, void* params);

// Real-Time Euler Functions
int realtime_euler_init(RealtimeEulerSolver* solver, size_t state_dim, double step_size,
                       DataCallback callback, void* callback_data);
void realtime_euler_free(RealtimeEulerSolver* solver);
double realtime_euler_step(RealtimeEulerSolver* solver, ODEFunction f, double t,
                          double* y, double h, void* params);

// Online Euler Functions
int online_euler_init(OnlineEulerSolver* solver, size_t state_dim, double initial_step_size,
                     double learning_rate);
void online_euler_free(OnlineEulerSolver* solver);
double online_euler_step(OnlineEulerSolver* solver, ODEFunction f, double t,
                        double* y, void* params);

// Dynamic Euler Functions
int dynamic_euler_init(DynamicEulerSolver* solver, size_t state_dim, double initial_step_size,
                      double adaptation_rate);
void dynamic_euler_free(DynamicEulerSolver* solver);
double dynamic_euler_step(DynamicEulerSolver* solver, ODEFunction f, double t,
                         double* y, void* params);

#ifdef __cplusplus
}
#endif

#endif /* REALTIME_ONLINE_H */
