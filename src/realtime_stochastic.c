/*
 * Real-Time and Stochastic RK3/AM Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "realtime_stochastic.h"
#include "rk3.h"
#include "adams.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Simple random number generator (linear congruential)
static uint32_t rng_state = 1;

static double random_uniform(void) {
    rng_state = rng_state * 1103515245 + 12345;
    return ((double)(rng_state & 0x7FFFFFFF)) / 2147483648.0;
}

static double random_gaussian(void) {
    // Box-Muller transform with safety checks
    double u1 = random_uniform();
    double u2 = random_uniform();
    
    // Ensure u1 is not zero or too small to avoid log(0)
    while (u1 <= 0.0 || u1 >= 1.0) {
        u1 = random_uniform();
    }
    while (u2 <= 0.0 || u2 >= 1.0) {
        u2 = random_uniform();
    }
    
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int realtime_rk3_init(RealtimeSolverState* state, size_t n, double h, size_t buffer_size) {
    if (!state || n == 0 || h <= 0) {
        return -1;
    }
    
    state->n = n;
    state->h = h;
    state->buffer_size = buffer_size;
    state->buffer_idx = 0;
    state->step_count = 0;
    state->t_current = 0.0;
    
    state->y_current = (double*)calloc(n, sizeof(double));
    state->derivatives = (double*)calloc(n, sizeof(double));
    
    if (buffer_size > 0) {
        state->y_buffer = (double*)calloc(n * buffer_size, sizeof(double));
        if (!state->y_buffer) {
            free(state->y_current);
            free(state->derivatives);
            return -1;
        }
    } else {
        state->y_buffer = NULL;
    }
    
    if (!state->y_current || !state->derivatives) {
        realtime_solver_free(state);
        return -1;
    }
    
    return 0;
}

int realtime_rk3_step(RealtimeSolverState* state,
                     void (*f)(double t, const double* y, double* dydt, void* params),
                     const double* y_new,
                     void* params,
                     RealtimeCallback callback,
                     void* user_data) {
    if (!state || !f || !y_new) {
        return -1;
    }
    
    // Update current state with new data
    memcpy(state->y_current, y_new, state->n * sizeof(double));
    
    // Perform RK3 step
    double t_new = rk3_step(f, state->t_current, state->y_current, state->n, state->h, params);
    
    // Update buffer if needed
    if (state->y_buffer && state->buffer_size > 0) {
        size_t idx = state->buffer_idx % state->buffer_size;
        memcpy(&state->y_buffer[idx * state->n], state->y_current, state->n * sizeof(double));
        state->buffer_idx++;
    }
    
    // Call callback if provided
    if (callback) {
        callback(t_new, state->y_current, state->n, user_data);
    }
    
    state->t_current = t_new;
    state->step_count++;
    
    return 0;
}

int realtime_adams_init(RealtimeSolverState* state, size_t n, double h, size_t buffer_size) {
    if (!state || n == 0 || h <= 0) {
        return -1;
    }
    
    state->n = n;
    state->h = h;
    state->buffer_size = buffer_size;
    state->buffer_idx = 0;
    state->step_count = 0;
    state->t_current = 0.0;
    state->history_count = 0;
    
    state->y_current = (double*)calloc(n, sizeof(double));
    // Adams 3rd order needs: f_n, f_n-1, f_n-2 (3 derivatives)
    state->derivatives = (double*)calloc(n * 3, sizeof(double));
    // Adams 3rd order needs: y_n, y_n-1, y_n-2 (3 states)
    state->y_history = (double*)calloc(n * 3, sizeof(double));
    // Time history: t_n, t_n-1, t_n-2
    state->t_history = (double*)calloc(3, sizeof(double));
    
    if (buffer_size > 0) {
        state->y_buffer = (double*)calloc(n * buffer_size, sizeof(double));
        if (!state->y_buffer) {
            free(state->y_current);
            free(state->derivatives);
            free(state->y_history);
            free(state->t_history);
            return -1;
        }
    } else {
        state->y_buffer = NULL;
    }
    
    if (!state->y_current || !state->derivatives || !state->y_history || !state->t_history) {
        realtime_solver_free(state);
        return -1;
    }
    
    return 0;
}

int realtime_adams_step(RealtimeSolverState* state,
                       void (*f)(double t, const double* y, double* dydt, void* params),
                       const double* y_new,
                       void* params,
                       RealtimeCallback callback,
                       void* user_data) {
    if (!state || !f || !y_new) {
        return -1;
    }
    
    // Update current state
    memcpy(state->y_current, y_new, state->n * sizeof(double));
    
    // Shift history: move y_n-1 -> y_n-2, y_n -> y_n-1, new -> y_n
    if (state->history_count >= 2) {
        // Shift states: y_n-2 = y_n-1, y_n-1 = y_n
        memcpy(&state->y_history[0 * state->n], &state->y_history[1 * state->n], state->n * sizeof(double));
        memcpy(&state->y_history[1 * state->n], &state->y_history[2 * state->n], state->n * sizeof(double));
        // Shift derivatives: f_n-2 = f_n-1, f_n-1 = f_n
        memcpy(&state->derivatives[0 * state->n], &state->derivatives[1 * state->n], state->n * sizeof(double));
        memcpy(&state->derivatives[1 * state->n], &state->derivatives[2 * state->n], state->n * sizeof(double));
        // Shift times: t_n-2 = t_n-1, t_n-1 = t_n
        state->t_history[0] = state->t_history[1];
        state->t_history[1] = state->t_history[2];
    }
    
    // Store current state and time in history
    memcpy(&state->y_history[2 * state->n], state->y_current, state->n * sizeof(double));
    state->t_history[2] = state->t_current;
    
    // Compute current derivative f_n = f(t_n, y_n)
    double* f_n = &state->derivatives[2 * state->n];
    f(state->t_current, state->y_current, f_n, params);
    
    // Increment history count (capped at 3 for 3rd order Adams)
    if (state->history_count < 3) {
        state->history_count++;
    }
    
    // Apply Adams-Bashforth 3rd order predictor
    if (state->history_count >= 3) {
        // Full Adams-Bashforth 3rd order: y_n+1 = y_n + h*(23*f_n - 16*f_n-1 + 5*f_n-2)/12
        double* f_n_1 = &state->derivatives[1 * state->n];  // f_n-1
        double* f_n_2 = &state->derivatives[0 * state->n];  // f_n-2
        double* y_n = &state->y_history[2 * state->n];      // y_n
        
        for (size_t i = 0; i < state->n; i++) {
            state->y_current[i] = y_n[i] + state->h * (23.0 * f_n[i] - 16.0 * f_n_1[i] + 5.0 * f_n_2[i]) / 12.0;
        }
        
        // Optional: Apply Adams-Moulton corrector
        double t_next = state->t_current + state->h;
        double* f_pred = (double*)malloc(state->n * sizeof(double));
        if (f_pred) {
            f(t_next, state->y_current, f_pred, params);
            // Adams-Moulton 3rd order: y_n+1 = y_n + h*(5*f_n+1 + 8*f_n - f_n-1)/12
            for (size_t i = 0; i < state->n; i++) {
                state->y_current[i] = y_n[i] + state->h * (5.0 * f_pred[i] + 8.0 * f_n[i] - f_n_1[i]) / 12.0;
            }
            free(f_pred);
        }
    } else if (state->history_count == 2) {
        // Use 2nd order Adams-Bashforth: y_n+1 = y_n + h*(3*f_n - f_n-1)/2
        double* f_n_1 = &state->derivatives[1 * state->n];
        double* y_n = &state->y_history[2 * state->n];
        for (size_t i = 0; i < state->n; i++) {
            state->y_current[i] = y_n[i] + state->h * (3.0 * f_n[i] - f_n_1[i]) / 2.0;
        }
    } else {
        // Use Euler for first step: y_n+1 = y_n + h*f_n
        for (size_t i = 0; i < state->n; i++) {
            state->y_current[i] = state->y_current[i] + state->h * f_n[i];
        }
    }
    
    // Update buffer
    if (state->y_buffer && state->buffer_size > 0) {
        size_t idx = state->buffer_idx % state->buffer_size;
        memcpy(&state->y_buffer[idx * state->n], state->y_current, state->n * sizeof(double));
        state->buffer_idx++;
    }
    
    // Call callback
    if (callback) {
        callback(state->t_current + state->h, state->y_current, state->n, user_data);
    }
    
    state->t_current += state->h;
    state->step_count++;
    
    return 0;
}

void realtime_solver_free(RealtimeSolverState* state) {
    if (!state) return;
    
    if (state->y_current) {
        free(state->y_current);
        state->y_current = NULL;
    }
    if (state->y_buffer) {
        free(state->y_buffer);
        state->y_buffer = NULL;
    }
    if (state->derivatives) {
        free(state->derivatives);
        state->derivatives = NULL;
    }
    if (state->y_history) {
        free(state->y_history);
        state->y_history = NULL;
    }
    if (state->t_history) {
        free(state->t_history);
        state->t_history = NULL;
    }
}

// Stochastic solver state
typedef struct {
    size_t n;
    double h;
    StochasticParams params;
    double* noise_buffer;
    double* brownian_path;
    size_t noise_idx;
    double last_noise_time;
} StochasticSolverState;

void* stochastic_rk3_init(size_t n, double h, const StochasticParams* params) {
    if (n == 0 || h <= 0 || !params) {
        return NULL;
    }
    
    StochasticSolverState* state = (StochasticSolverState*)calloc(1, sizeof(StochasticSolverState));
    if (!state) {
        return NULL;
    }
    
    state->n = n;
    state->h = h;
    state->params = *params;
    state->noise_idx = 0;
    state->last_noise_time = 0.0;
    
    // Initialize random seed
    if (params->seed == 0) {
        rng_state = (uint32_t)time(NULL);
    } else {
        rng_state = (uint32_t)(params->seed * 1000000);
    }
    
    // Allocate noise buffer
    size_t buffer_size = (params->use_brownian) ? 1000 : 100;
    state->noise_buffer = (double*)calloc(buffer_size, sizeof(double));
    if (!state->noise_buffer) {
        free(state);
        return NULL;
    }
    
    if (params->use_brownian) {
        state->brownian_path = (double*)calloc(n, sizeof(double));
        if (!state->brownian_path) {
            free(state->noise_buffer);
            free(state);
            return NULL;
        }
    } else {
        state->brownian_path = NULL;
    }
    
    return state;
}

double stochastic_rk3_step(void* solver,
                           void (*f)(double t, const double* y, double* dydt, void* params),
                           double t0,
                           double* y0,
                           void* params) {
    if (!solver || !f || !y0) {
        return t0;
    }
    
    StochasticSolverState* state = (StochasticSolverState*)solver;
    
    // Generate stochastic noise
    double noise_scale = state->params.noise_amplitude * sqrt(state->h);
    double* noise = (double*)calloc(state->n, sizeof(double));
    if (!noise) {
        return t0;
    }
    
    if (state->params.use_brownian) {
        // Brownian motion: dW = sqrt(dt) * N(0,1)
        if (!state->brownian_path) {
            free(noise);
            return t0;
        }
        for (size_t i = 0; i < state->n; i++) {
            double dW = random_gaussian() * sqrt(state->h);
            state->brownian_path[i] += dW;
            noise[i] = state->params.noise_amplitude * state->brownian_path[i];
        }
    } else {
        // White noise
        for (size_t i = 0; i < state->n; i++) {
            noise[i] = noise_scale * random_gaussian();
        }
    }
    
    // Perform RK3 step
    double t_new = rk3_step(f, t0, y0, state->n, state->h, params);
    
    // Add stochastic term
    for (size_t i = 0; i < state->n; i++) {
        y0[i] += noise[i];
    }
    
    free(noise);
    return t_new;
}

void* stochastic_adams_init(size_t n, double h, const StochasticParams* params) {
    return stochastic_rk3_init(n, h, params); // Same initialization
}

double stochastic_adams_step(void* solver,
                            void (*f)(double t, const double* y, double* dydt, void* params),
                            double t0,
                            double* y0,
                            void* params) {
    if (!solver || !f || !y0) {
        return t0;
    }
    
    StochasticSolverState* state = (StochasticSolverState*)solver;
    
    // Generate stochastic noise
    double noise_scale = state->params.noise_amplitude * sqrt(state->h);
    double* noise = (double*)calloc(state->n, sizeof(double));
    if (!noise) {
        return t0;
    }
    
    if (state->params.use_brownian) {
        for (size_t i = 0; i < state->n; i++) {
            double dW = random_gaussian() * sqrt(state->h);
            state->brownian_path[i] += dW;
            noise[i] = state->params.noise_amplitude * state->brownian_path[i];
        }
    } else {
        for (size_t i = 0; i < state->n; i++) {
            noise[i] = noise_scale * random_gaussian();
        }
    }
    
    // Compute derivative
    double* dydt = (double*)calloc(state->n, sizeof(double));
    if (!dydt) {
        free(noise);
        return t0;
    }
    f(t0, y0, dydt, params);
    
    // Simple Adams step with noise
    for (size_t i = 0; i < state->n; i++) {
        y0[i] = y0[i] + state->h * dydt[i] + noise[i];
    }
    
    free(noise);
    free(dydt);
    return t0 + state->h;
}

void stochastic_solver_free(void* solver) {
    if (!solver) return;
    
    StochasticSolverState* state = (StochasticSolverState*)solver;
    
    if (state->noise_buffer) {
        free(state->noise_buffer);
    }
    if (state->brownian_path) {
        free(state->brownian_path);
    }
    free(state);
}

double data_driven_adaptive_step(const double* error_history,
                                 size_t history_size,
                                 double current_h,
                                 double target_error) {
    if (!error_history || history_size == 0 || current_h <= 0 || target_error <= 0) {
        return current_h;
    }
    
    // Compute average error
    double avg_error = 0.0;
    for (size_t i = 0; i < history_size; i++) {
        avg_error += error_history[i];
    }
    avg_error /= history_size;
    
    // Adaptive step size: h_new = h * (target_error / avg_error)^(1/order)
    // For RK3, order is 3
    double ratio = target_error / (avg_error + 1e-10);
    double factor = pow(ratio, 1.0 / 3.0);
    
    // Limit factor to reasonable range
    if (factor < 0.1) factor = 0.1;
    if (factor > 2.0) factor = 2.0;
    
    return current_h * factor;
}

int data_driven_method_select(double stiffness_estimate,
                              double error_tolerance,
                              double speed_requirement) {
    // Simple heuristic: use Adams for stiff systems, RK3 for others
    // If speed is critical, prefer Adams (typically faster)
    
    if (stiffness_estimate > 100.0) {
        // Very stiff system - use Adams
        return 1;
    } else if (speed_requirement > 1000000.0) {
        // High speed requirement - use Adams
        return 1;
    } else if (error_tolerance < 1e-6) {
        // High accuracy requirement - use RK3
        return 0;
    } else {
        // Balanced - can use either
        return -1;
    }
}
