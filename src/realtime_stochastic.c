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

static double random_uniform() {
    rng_state = rng_state * 1103515245 + 12345;
    return ((double)(rng_state & 0x7FFFFFFF)) / 2147483648.0;
}

static double random_gaussian() {
    // Box-Muller transform
    double u1 = random_uniform();
    double u2 = random_uniform();
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
    
    state->y_current = (double*)calloc(n, sizeof(double));
    state->derivatives = (double*)calloc(n * 3, sizeof(double)); // Need history for Adams
    
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
    
    // Compute derivative
    double* dydt = &state->derivatives[0];
    f(state->t_current, state->y_current, dydt, params);
    
    // Simple Adams-Bashforth 3rd order (simplified for real-time)
    // In full implementation, would use history buffer
    double* y_prev = state->y_current;
    double* f_prev = dydt;
    
    // Predictor step
    for (size_t i = 0; i < state->n; i++) {
        state->y_current[i] = y_prev[i] + state->h * f_prev[i];
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
    state->brownian_path = (params->use_brownian) ? (double*)calloc(n, sizeof(double)) : NULL;
    
    if (!state->noise_buffer || (params->use_brownian && !state->brownian_path)) {
        stochastic_solver_free(state);
        return NULL;
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
