/*
 * Real-Time and Stochastic RK3/AM Solvers
 * Data-Driven Methods for Differential Equations
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef REALTIME_STOCHASTIC_H
#define REALTIME_STOCHASTIC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * Real-time solver state
 */
typedef struct {
    double* y_current;      // Current state
    double* y_buffer;       // Buffer for streaming data
    double* derivatives;   // Derivative buffer
    size_t n;              // System dimension
    double t_current;      // Current time
    double h;              // Step size
    size_t buffer_size;    // Buffer size for streaming
    size_t buffer_idx;     // Current buffer index
    uint64_t step_count;   // Total steps processed
} RealtimeSolverState;

/**
 * Stochastic solver parameters
 */
typedef struct {
    double noise_amplitude;     // Amplitude of stochastic noise
    double noise_correlation;   // Correlation time for noise
    int use_brownian;          // Use Brownian motion (1) or white noise (0)
    double seed;               // Random seed (0 = auto)
} StochasticParams;

/**
 * Real-time callback function type
 * Called when new data point is available
 */
typedef void (*RealtimeCallback)(double t, const double* y, size_t n, void* user_data);

/**
 * Initialize real-time RK3 solver
 * 
 * @param state: Solver state structure
 * @param n: System dimension
 * @param h: Step size
 * @param buffer_size: Buffer size for streaming (0 = no buffering)
 * @return: 0 on success, -1 on failure
 */
int realtime_rk3_init(RealtimeSolverState* state, size_t n, double h, size_t buffer_size);

/**
 * Step real-time RK3 solver with streaming data
 * 
 * @param state: Solver state
 * @param f: ODE function
 * @param y_new: New state data (input)
 * @param params: ODE parameters
 * @param callback: Optional callback for each step
 * @param user_data: User data for callback
 * @return: 0 on success, -1 on failure
 */
int realtime_rk3_step(RealtimeSolverState* state,
                     void (*f)(double t, const double* y, double* dydt, void* params),
                     const double* y_new,
                     void* params,
                     RealtimeCallback callback,
                     void* user_data);

/**
 * Initialize real-time Adams solver
 */
int realtime_adams_init(RealtimeSolverState* state, size_t n, double h, size_t buffer_size);

/**
 * Step real-time Adams solver
 */
int realtime_adams_step(RealtimeSolverState* state,
                       void (*f)(double t, const double* y, double* dydt, void* params),
                       const double* y_new,
                       void* params,
                       RealtimeCallback callback,
                       void* user_data);

/**
 * Free real-time solver resources
 */
void realtime_solver_free(RealtimeSolverState* state);

/**
 * Initialize stochastic RK3 solver
 * 
 * @param n: System dimension
 * @param h: Step size
 * @param params: Stochastic parameters
 * @return: Solver state pointer, NULL on failure
 */
void* stochastic_rk3_init(size_t n, double h, const StochasticParams* params);

/**
 * Step stochastic RK3 solver
 * 
 * @param solver: Solver state
 * @param f: ODE function
 * @param t0: Current time
 * @param y0: Current state (modified in-place)
 * @param params: ODE parameters
 * @return: New time (t0 + h)
 */
double stochastic_rk3_step(void* solver,
                           void (*f)(double t, const double* y, double* dydt, void* params),
                           double t0,
                           double* y0,
                           void* params);

/**
 * Initialize stochastic Adams solver
 */
void* stochastic_adams_init(size_t n, double h, const StochasticParams* params);

/**
 * Step stochastic Adams solver
 */
double stochastic_adams_step(void* solver,
                            void (*f)(double t, const double* y, double* dydt, void* params),
                            double t0,
                            double* y0,
                            void* params);

/**
 * Free stochastic solver resources
 */
void stochastic_solver_free(void* solver);

/**
 * Data-driven adaptive step size control
 * Uses historical error data to adjust step size
 * 
 * @param error_history: Array of recent errors
 * @param history_size: Size of error history
 * @param current_h: Current step size
 * @param target_error: Target error tolerance
 * @return: Recommended new step size
 */
double data_driven_adaptive_step(const double* error_history,
                                 size_t history_size,
                                 double current_h,
                                 double target_error);

/**
 * Data-driven method selection
 * Selects best method (RK3 vs AM) based on system characteristics
 * 
 * @param stiffness_estimate: Estimated stiffness of system
 * @param error_tolerance: Error tolerance
 * @param speed_requirement: Speed requirement (steps/sec)
 * @return: 0 = RK3, 1 = Adams, -1 = use both
 */
int data_driven_method_select(double stiffness_estimate,
                              double error_tolerance,
                              double speed_requirement);

#ifdef __cplusplus
}
#endif

#endif /* REALTIME_STOCHASTIC_H */
